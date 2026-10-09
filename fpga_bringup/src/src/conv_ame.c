#include <animegan/ame.h>
#include <animegan/bf16.h>
#include <animegan/platform.h>
#include <animegan/runtime.h>

/* CONV_2D on the AME matrix unit as an implicit GEMM over the channel-blocked
 * layout: no im2col, no weight packing, no zero padding at run time.
 *   one tile = TM consecutive output pixels of one output row x KO channels,
 *   TM = min(128, OW); tiles are taken in groups of three (the last group has
 *   one or two), and for every (ky, kx, channel block) the group shares one B:
 *     B tile   = weights [(ky,kx,cb)][KO][32] -> tr0, row stride 64 B
 *                (tile-major, contiguous)
 *     A tile i = the TM input pixels (oy*sh+ky, ox*sw+kx) of block cb, 64 B
 *                each -> tr1..tr3, read straight from the padded input with
 *                row stride sw*64 B (contiguous 8 KB for stride 1)
 *     acc_i += A_i . B^T (fp32, acc0..acc2)
 *   then msce32 stores each fp32 [TM][KO] block to its own scratch slot and the
 *   epilogue adds the bias, applies the fused LeakyReLU, rounds to bf16 and
 *   writes the output pixels (channel block by channel block, 64 B per pixel).
 * Weight lanes beyond C and channels beyond KO are zero, so the first layer
 * (C = 3) and the last (KO = 3) need no special case. A shape with OW not a
 * multiple of TM or KO > 128 returns AG_AME_UNSUPPORTED.
 * K packing (first layer, AG_TENSOR_LANE_PACK input + options.integers[6]):
 * input lane (dy*KW+dx)*C+c of pixel (y, x) holds pixel (y+dy, x+dx), so step
 * s = 0..S-1 takes A at input row oy+s*R, kx = 0, and B = block s of the
 * repacked weights: S = ceil(KH/R) steps instead of KH*KW.
 * Overlap: the scratch holds two sets of CONV_GROUP C blocks. While the matrix
 * unit runs group g (its C goes to set g & 1), the epilogue of group g-1 runs
 * in slices of ceil(rows / steps) output rows after each matrix step's asm
 * block; ag_conv_overlap = 0 runs each group's epilogue right after its C
 * store instead.
 * The matrix unit addresses the input and the scratch through ag_plat_ame_addr
 * (the board's DDR view); fences order the RVV writes of the input before the
 * loads, the C stores before the epilogue reads them, and (after each group's
 * step loop) the epilogue reads of a C set before the stores that reuse it. */
#define CONV_GROUP 3

typedef char ag_conv_scratch_check[AG_CONV_AME_SCRATCH == (size_t)2 * CONV_GROUP * AG_AME_TILE_M * AG_AME_TILE_N * sizeof(float) ? 1 : -1];

int ag_conv_overlap = 1;

static int conv_tile_rows(const ag_conv_geom *g) { return g->out.W < AG_AME_TILE_M ? g->out.W : AG_AME_TILE_M; }

/* bias, fused LeakyReLU, bf16, blocked output for the tile at (oy, m0); e: RVV
 * epilogue state prepared once per node (unused by the scalar epilogue) */
static void conv_epilogue(const ag_conv_geom *g, const ag_conv_epi *e, int32_t oy, int32_t m0, int32_t TM, const float *cbuf) {
#ifdef AG_RVV
    ag_conv_epilogue_rvv(e, g->y + ((size_t)g->out.origin + (size_t)oy * g->out.pitch + m0) * AG_BLOCK, cbuf, TM);
#else
    int32_t cb, r, j, ko;
    (void)e;
    for (cb = 0; cb < g->out.blocks; ++cb) {
        uint16_t *yp = g->y + ((size_t)cb * g->out.plane + g->out.origin + (size_t)oy * g->out.pitch + m0) * AG_BLOCK;
        for (r = 0; r < TM; ++r) {
            const float *crow = cbuf + (size_t)r * g->KO;
            uint16_t *yrow = yp + (size_t)r * AG_BLOCK;
            for (j = 0; j < AG_BLOCK; ++j) {
                float v = 0.0f;
                ko = cb * AG_BLOCK + j;
                if (ko < g->KO) {
                    v = crow[ko] + (g->bias ? g->bias[ko] : 0.0f);
                    if (g->fused == AG_FUSED_LEAKY && v < 0.0f) v *= g->alpha;
                }
                yrow[j] = ag_f32_to_bf16(v);
            }
        }
    }
#endif
}

/* epilogue still to run for one stored group: tiles t0..t0+n-1 with C at c,
 * next row `row` of tile `tile` */
typedef struct {
    const float *c;
    int32_t t0, n, tile, row;
} conv_pending;

/* run up to `rows` output rows of the pending epilogue */
static void conv_drain(const ag_conv_geom *g, const ag_conv_epi *e, conv_pending *p, int32_t nm, int32_t TM, int32_t rows) {
    size_t cblock = (size_t)TM * g->KO;
    while (rows > 0 && p->tile < p->n) {
        int32_t t = p->t0 + p->tile, cnt = TM - p->row < rows ? TM - p->row : rows;
        conv_epilogue(g, e, t / nm, t % nm * TM + p->row, cnt, p->c + p->tile * cblock + (size_t)p->row * g->KO);
        rows -= cnt;
        if ((p->row += cnt) == TM) { p->row = 0; ++p->tile; }
    }
}

int ag_conv2d_ame(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                  void *const tensor_data[], void *scratch, size_t scratch_size) {
    ag_conv_geom g;
    ag_conv_epi epi;
    conv_pending pend = {0, 0, 0, 0, 0};
    float *cbuf = scratch, *cmat;
    const uint16_t *xmat, *xa[CONV_GROUP], *wb;
    int32_t TM, nm, tiles, t0, n, i, ky, kx, cb, KH, KW, CB, ky_rows, slice, set, pk = node->options.integers[6];
    long a_stride, c_stride;
    size_t cblock;
    int rc = ag_conv2d_check(model, node, inputs, outputs, tensor_data, &g);
    if (rc) return rc;
    TM = conv_tile_rows(&g);
    if (g.out.W % TM || g.KO > AG_AME_TILE_N) return AG_AME_UNSUPPORTED;
    if (!scratch || scratch_size < AG_CONV_AME_SCRATCH) return -29;
    /* loop bounds: (ky, kx, cb) over the weights, or S packed steps of R input rows */
    KH = g.KH; KW = g.KW; CB = g.CB; ky_rows = 1; wb = g.w;
    if ((model->tensors[inputs[0]].flags & AG_TENSOR_LANE_PACK) && pk > 0 && (uint32_t)pk <= model->tensor_count && tensor_data[pk - 1]) {
        const ag_tensor_desc *xt = model->tensors + inputs[0], *pt = model->tensors + pk - 1;
        int32_t R = (int32_t)(xt->reserved & 0xFFu);
        if (R < 1 || (int32_t)(xt->reserved >> 8 & 0xFFu) != g.KW || R * g.KW * g.C > AG_BLOCK || g.CB != 1 || g.stride_h != 1 ||
            g.stride_w != 1 || pt->type != AG_TYPE_BF16 || !pt->block || pt->shape[0] != g.KO || pt->shape[1] != (g.KH + R - 1) / R ||
            pt->shape[2] != 1 || pt->shape[3] != R * g.KW * g.C) return -28;
        KH = pt->shape[1]; KW = 1; CB = 1; ky_rows = R; wb = tensor_data[pk - 1];
    }
#ifdef AG_RVV
    ag_conv_epilogue_rvv_init(&epi, &g);
#endif
    nm = g.out.W / TM;
    tiles = g.out.H * nm;
    a_stride = (long)g.stride_w * AG_BLOCK * 2;
    c_stride = (long)g.KO * 4;
    cblock = (size_t)TM * g.KO;
    xmat = ag_plat_ame_addr(g.x);
    cmat = ag_plat_ame_addr(cbuf);
    ag_plat_ame_sync_in(g.x, (size_t)g.in.blocks * g.in.plane * AG_BLOCK * 2);
    ag_ame_rne();
    ag_ame_set_tile(TM, g.KO, AG_AME_TILE_K);
    for (t0 = 0, set = 0; t0 < tiles; t0 += n, set ^= 1) {
        float *cs = cbuf + (size_t)set * CONV_GROUP * cblock, *cm = cmat + (size_t)set * CONV_GROUP * cblock;
        n = tiles - t0 < CONV_GROUP ? tiles - t0 : CONV_GROUP;
        for (i = 0; i < n; ++i) {
            int32_t oy = (t0 + i) / nm, m0 = (t0 + i) % nm * TM;
            xa[i] = xmat + ((size_t)g.in.origin + (size_t)oy * g.stride_h * g.in.pitch + (size_t)m0 * g.stride_w) * AG_BLOCK;
        }
        /* rows of the previous group's epilogue per matrix step */
        slice = pend.tile < pend.n ? ((pend.n - pend.tile) * TM - pend.row + KH * KW * CB - 1) / (KH * KW * CB) : 0;
        AG_AME_ZERO("acc0");
        AG_AME_ZERO("acc1");
        AG_AME_ZERO("acc2");
        for (ky = 0; ky < KH; ++ky) {
            for (kx = 0; kx < KW; ++kx) {
                for (cb = 0; cb < CB; ++cb) {
                    size_t off = ((size_t)cb * g.in.plane + (size_t)ky * ky_rows * g.in.pitch + kx) * AG_BLOCK;
                    const uint16_t *b = wb + ((size_t)(ky * KW + kx) * CB + cb) * g.KO * AG_BLOCK;
                    if (n == 3) ag_ame_macc3(b, AG_BLOCK * 2, xa[0] + off, xa[1] + off, xa[2] + off, a_stride);
                    else if (n == 2) ag_ame_macc2(b, AG_BLOCK * 2, xa[0] + off, xa[1] + off, a_stride);
                    else ag_ame_macc1(b, AG_BLOCK * 2, xa[0] + off, a_stride);
                    if (slice) conv_drain(&g, &epi, &pend, nm, TM, slice);
                }
            }
        }
        conv_drain(&g, &epi, &pend, nm, TM, INT32_MAX);
        ag_plat_ame_sync_in(cs, (size_t)n * cblock * sizeof(float));   /* epilogue reads of the other set before later C stores into it */
        AG_AME_STORE_C("acc0", cm, c_stride);
        if (n > 1) AG_AME_STORE_C("acc1", cm + cblock, c_stride);
        if (n > 2) AG_AME_STORE_C("acc2", cm + 2 * cblock, c_stride);
        ag_plat_ame_sync_out(cs, (size_t)n * cblock * sizeof(float));
        pend.c = cs; pend.t0 = t0; pend.n = n; pend.tile = 0; pend.row = 0;
        if (!ag_conv_overlap) conv_drain(&g, &epi, &pend, nm, TM, INT32_MAX);
    }
    conv_drain(&g, &epi, &pend, nm, TM, INT32_MAX);
    ag_plat_ame_release();
    return 0;
}
