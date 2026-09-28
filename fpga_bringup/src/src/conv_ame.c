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
 * Channels beyond C / KO are zero lanes in both operands, so the first layer
 * (C = 3) and the last (KO = 3) need no special case. A shape with OW not a
 * multiple of TM or KO > 128 returns AG_AME_UNSUPPORTED.
 * The matrix unit addresses the input and the scratch through ag_plat_ame_addr
 * (the board's DDR view); fences order the RVV writes of the input before the
 * loads and the C stores before the epilogue reads them. */
#define CONV_GROUP 3

static int conv_tile_rows(const ag_conv_geom *g) { return g->out.W < AG_AME_TILE_M ? g->out.W : AG_AME_TILE_M; }

size_t ag_conv2d_ame_scratch(const ag_model *model, const ag_node_desc *node) {
    (void)model; (void)node;
    return (size_t)CONV_GROUP * AG_AME_TILE_M * AG_AME_TILE_N * sizeof(float);
}

/* bias, fused LeakyReLU, bf16, blocked output for the tile at (oy, m0) */
static void conv_epilogue(const ag_conv_geom *g, int32_t oy, int32_t m0, int32_t TM, const float *cbuf) {
#ifdef AG_RVV
    ag_conv_epilogue_rvv(g->y + ((size_t)g->out.origin + (size_t)oy * g->out.pitch + m0) * AG_BLOCK, (size_t)g->out.plane * AG_BLOCK,
                         cbuf, TM, g->KO, g->out.blocks, g->bias, g->fused, g->alpha);
#else
    int32_t cb, r, j, ko;
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

int ag_conv2d_ame(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                  void *const tensor_data[], void *scratch, size_t scratch_size) {
    ag_conv_geom g;
    float *cbuf = scratch, *cmat;
    const uint16_t *xmat, *xa[CONV_GROUP];
    int32_t TM, nm, tiles, t0, n, i, ky, kx, cb;
    long a_stride, c_stride;
    size_t cblock;
    int rc = ag_conv2d_check(model, node, inputs, outputs, tensor_data, &g);
    if (rc) return rc;
    TM = conv_tile_rows(&g);
    if (g.out.W % TM || g.KO > AG_AME_TILE_N) return AG_AME_UNSUPPORTED;
    if (!scratch || scratch_size < ag_conv2d_ame_scratch(model, node)) return -29;
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
    for (t0 = 0; t0 < tiles; t0 += n) {
        n = tiles - t0 < CONV_GROUP ? tiles - t0 : CONV_GROUP;
        for (i = 0; i < n; ++i) {
            int32_t oy = (t0 + i) / nm, m0 = (t0 + i) % nm * TM;
            xa[i] = xmat + ((size_t)g.in.origin + (size_t)oy * g.stride_h * g.in.pitch + (size_t)m0 * g.stride_w) * AG_BLOCK;
        }
        AG_AME_ZERO("acc0");
        AG_AME_ZERO("acc1");
        AG_AME_ZERO("acc2");
        for (ky = 0; ky < g.KH; ++ky) {
            for (kx = 0; kx < g.KW; ++kx) {
                for (cb = 0; cb < g.CB; ++cb) {
                    size_t off = ((size_t)cb * g.in.plane + (size_t)ky * g.in.pitch + kx) * AG_BLOCK;
                    const uint16_t *b = g.w + ((size_t)(ky * g.KW + kx) * g.CB + cb) * g.KO * AG_BLOCK;
                    if (n == 3) ag_ame_macc3(b, AG_BLOCK * 2, xa[0] + off, xa[1] + off, xa[2] + off, a_stride);
                    else if (n == 2) ag_ame_macc2(b, AG_BLOCK * 2, xa[0] + off, xa[1] + off, a_stride);
                    else ag_ame_macc1(b, AG_BLOCK * 2, xa[0] + off, a_stride);
                }
            }
        }
        AG_AME_STORE_C("acc0", cmat, c_stride);
        if (n > 1) AG_AME_STORE_C("acc1", cmat + cblock, c_stride);
        if (n > 2) AG_AME_STORE_C("acc2", cmat + 2 * cblock, c_stride);
        ag_plat_ame_sync_out(cbuf, (size_t)n * cblock * sizeof(float));
        for (i = 0; i < n; ++i) conv_epilogue(&g, (t0 + i) / nm, (t0 + i) % nm * TM, TM, cbuf + i * cblock);
    }
    ag_plat_ame_release();
    return 0;
}
