#include <animegan/bf16.h>
#include <animegan/runtime.h>
#include <riscv_vector.h>
#include <string.h>

/* RVV kernels of the bf16 runtime (AG_RVV builds): ADD, LEAKY_RELU,
 * RESIZE_BILINEAR, TANH, LADE, MIRROR_PAD (blocked and dense-input),
 * the convolution epilogue and the UINT8 <-> bf16 frame conversions of the
 * board program. bf16 <-> fp32 is done with integer shifts (no zvfbfmin
 * needed) and every kernel uses the same fp32 formulas as the scalar kernel
 * in src/ops_bf16.c, so host and board agree except for the summation order
 * inside LADE's statistics. Blocked tensors are processed row by row: one row
 * of one channel block is W x 32 contiguous bf16.
 *
 * Vector shapes: bf16 at e16 LMUL=2 widened to fp32 at LMUL=4 (128 elements
 * per strip at VLEN=1024 = 4 pixels of a channel block); the per-pixel
 * kernels (resize) use LMUL=1 fp32 = one 32-lane pixel. */

static inline vfloat32m4_t load4(const uint16_t *p, size_t vl) {
    vuint32m4_t u = __riscv_vzext_vf2_u32m4(__riscv_vle16_v_u16m2(p, vl), vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(u, 16, vl));
}
static inline vuint16m2_t bf16_4(vfloat32m4_t f, size_t vl) {   /* round to nearest even */
    vuint32m4_t u = __riscv_vreinterpret_v_f32m4_u32m4(f);
    vuint32m4_t lsb = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(u, 16, vl), 1, vl);
    u = __riscv_vadd_vv_u32m4(__riscv_vadd_vx_u32m4(u, 0x7FFFu, vl), lsb, vl);
    return __riscv_vnsrl_wx_u16m2(u, 16, vl);
}
static inline void store4(uint16_t *p, vfloat32m4_t f, size_t vl) { __riscv_vse16_v_u16m2(p, bf16_4(f, vl), vl); }

static inline vfloat32m1_t load1(const uint16_t *p, size_t vl) {
    vuint32m1_t u = __riscv_vzext_vf2_u32m1(__riscv_vle16_v_u16mf2(p, vl), vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vsll_vx_u32m1(u, 16, vl));
}
static inline void store1(uint16_t *p, vfloat32m1_t f, size_t vl) {
    vuint32m1_t u = __riscv_vreinterpret_v_f32m1_u32m1(f);
    vuint32m1_t lsb = __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(u, 16, vl), 1, vl);
    u = __riscv_vadd_vv_u32m1(__riscv_vadd_vx_u32m1(u, 0x7FFFu, vl), lsb, vl);
    __riscv_vse16_v_u16mf2(p, __riscv_vnsrl_wx_u16mf2(u, 16, vl), vl);
}

static inline vfloat32m4_t leaky4(vfloat32m4_t v, int fused, float alpha, size_t vl) {
    vbool8_t neg;
    if (fused != AG_FUSED_LEAKY) return v;
    neg = __riscv_vmflt_vf_f32m4_b8(v, 0.0f, vl);
    return __riscv_vmerge_vvm_f32m4(v, __riscv_vfmul_vf_f32m4(v, alpha, vl), neg, vl);
}

/* interior row y of channel block cb */
static inline uint16_t *row_ptr(uint16_t *base, const ag_view *v, int32_t cb, int32_t y) {
    return base + ((size_t)cb * v->plane + v->origin + (size_t)y * v->pitch) * AG_BLOCK;
}

static int act(const ag_model *model, int32_t id, void *const tensor_data[], ag_view *v, uint16_t **p) {
    const ag_tensor_desc *t;
    if (id < 0 || (uint32_t)id >= model->tensor_count || !tensor_data[id]) return -1;
    t = model->tensors + id;
    if (t->rank != 4 || t->type != AG_TYPE_BF16) return -2;
    *v = ag_view_of(t);
    *p = tensor_data[id];
    return 0;
}

static int same_geometry(const ag_view *a, const ag_view *b) { return a->block && b->block && a->H == b->H && a->W == b->W && a->C == b->C; }

/* ---- ADD (+ fused LeakyReLU) and LEAKY_RELU: row-wise elementwise ---- */
static int elementwise_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                           void *const tensor_data[], int add) {
    ag_view va, vb, vo;
    uint16_t *a, *b = 0, *y;
    int32_t cb, r;
    int fused = add ? node->options.integers[0] : AG_FUSED_LEAKY;
    float alpha = node->options.floats[0];
    if (act(model, inputs[0], tensor_data, &va, &a) || act(model, outputs[0], tensor_data, &vo, &y) || !same_geometry(&va, &vo)) return AG_AME_UNSUPPORTED;
    if (add && (node->input_count != 2 || act(model, inputs[1], tensor_data, &vb, &b) || !same_geometry(&vb, &vo))) return AG_AME_UNSUPPORTED;
    for (cb = 0; cb < vo.blocks; ++cb)
        for (r = 0; r < vo.H; ++r) {
            const uint16_t *pa = row_ptr(a, &va, cb, r), *pb = add ? row_ptr(b, &vb, cb, r) : 0;
            uint16_t *py = row_ptr(y, &vo, cb, r);
            size_t n = (size_t)vo.W * AG_BLOCK, i = 0;
            while (i < n) {
                size_t vl = __riscv_vsetvl_e16m2(n - i);
                vfloat32m4_t s = load4(pa + i, vl);
                if (add) s = __riscv_vfadd_vv_f32m4(s, load4(pb + i, vl), vl);
                store4(py + i, leaky4(s, fused, alpha, vl), vl);
                i += vl;
            }
        }
    return 0;
}

/* ---- RESIZE_BILINEAR x2, asymmetric ----
 * even output rows = horizontal expansion of input row i, one pixel at a time
 * (out[2j] = in[j], out[2j+1] = in[j] + (in[j+1] - in[j]) * 0.5, last column
 * clamped); odd rows = even row 2i + (even row 2i+2 - even row 2i) * 0.5 with
 * the last odd row equal to its even row: the scalar kernel's formulas. */
static int resize_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t cb, r, j;
    if (node->options.integers[0] || node->options.integers[1]) return AG_AME_UNSUPPORTED;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return AG_AME_UNSUPPORTED;
    if (!vo.block || !vi.block || vi.C != vo.C || vo.H != 2 * vi.H || vo.W != 2 * vi.W) return AG_AME_UNSUPPORTED;
    if (__riscv_vsetvl_e32m1(AG_BLOCK) != AG_BLOCK) return AG_AME_UNSUPPORTED;   /* one pixel per m1 register needs VLEN >= 1024 */
    for (cb = 0; cb < vo.blocks; ++cb) {
        for (r = 0; r < vi.H; ++r) {
            const uint16_t *px = row_ptr(x, &vi, cb, r);
            uint16_t *py = row_ptr(y, &vo, cb, 2 * r);
            size_t vl = __riscv_vsetvl_e32m1(AG_BLOCK);
            vfloat32m1_t a = load1(px, vl);
            for (j = 0; j < vi.W; ++j) {
                vfloat32m1_t b = j + 1 < vi.W ? load1(px + (size_t)(j + 1) * AG_BLOCK, vl) : a;
                vfloat32m1_t h = __riscv_vfadd_vv_f32m1(a, __riscv_vfmul_vf_f32m1(__riscv_vfsub_vv_f32m1(b, a, vl), 0.5f, vl), vl);
                store1(py + (size_t)(2 * j) * AG_BLOCK, a, vl);
                store1(py + (size_t)(2 * j + 1) * AG_BLOCK, h, vl);
                a = b;
            }
        }
        for (r = 0; r < vi.H; ++r) {
            const uint16_t *top = row_ptr(y, &vo, cb, 2 * r), *bottom = row_ptr(y, &vo, cb, r + 1 < vi.H ? 2 * r + 2 : 2 * r);
            uint16_t *py = row_ptr(y, &vo, cb, 2 * r + 1);
            size_t n = (size_t)vo.W * AG_BLOCK, i = 0;
            while (i < n) {
                size_t vl = __riscv_vsetvl_e16m2(n - i);
                vfloat32m4_t t = load4(top + i, vl), d = __riscv_vfsub_vv_f32m4(load4(bottom + i, vl), t, vl);
                store4(py + i, __riscv_vfadd_vv_f32m4(t, __riscv_vfmul_vf_f32m4(d, 0.5f, vl), vl), vl);
                i += vl;
            }
        }
    }
    return 0;
}

/* ---- TANH: exp by range reduction, same polynomial as ag_tanhf ---- */
static inline vfloat32m4_t exp4(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t n = __riscv_vfcvt_f_x_v_f32m4(__riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(x, 1.44269504f, vl), vl), vl);
    vfloat32m4_t f = __riscv_vfsub_vv_f32m4(x, __riscv_vfmul_vf_f32m4(n, 0.69314718f, vl), vl), p;
    vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(__riscv_vfcvt_x_f_v_i32m4(n, vl), 127, vl), 23, vl);
    p = __riscv_vfmv_v_f_f32m4(1.0f / 720.0f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f / 120.0f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f / 24.0f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f / 6.0f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 0.5f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}

static inline vfloat32m4_t tanh4(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t c = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(x, -9.0f, vl), 9.0f, vl);
    vfloat32m4_t e = exp4(__riscv_vfmul_vf_f32m4(c, 2.0f, vl), vl);
    vfloat32m4_t t = __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(e, 1.0f, vl), 2.0f, vl), 1.0f, vl);
    vbool8_t hi = __riscv_vmfgt_vf_f32m4_b8(x, 9.0f, vl), lo = __riscv_vmflt_vf_f32m4_b8(x, -9.0f, vl);
    t = __riscv_vfmerge_vfm_f32m4(t, 1.0f, hi, vl);
    return __riscv_vfmerge_vfm_f32m4(t, -1.0f, lo, vl);
}

/* blocked input -> dense NHWC output, unit-stride only: a strip of P pixels
 * (P x 32 lanes at e16 LMUL=8, P = 16 at VLEN=1024) is loaded, vcompress keeps
 * lanes < C (P x C packed values), tanh runs on those and one store writes
 * P x C dense elements */
static int tanh_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t r;
    size_t vlmax = __riscv_vsetvlmax_e16m8(), P = vlmax / AG_BLOCK;
    vbool2_t keep;
    (void)node;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return AG_AME_UNSUPPORTED;
    if (!vi.block || vo.block || vi.H != vo.H || vi.W != vo.W || vi.C != vo.C || vi.C > AG_BLOCK) return AG_AME_UNSUPPORTED;
    if (!P || P * (size_t)vo.C > __riscv_vsetvlmax_e16m2()) return AG_AME_UNSUPPORTED;
    keep = __riscv_vmsltu_vx_u16m8_b2(__riscv_vand_vx_u16m8(__riscv_vid_v_u16m8(vlmax), AG_BLOCK - 1, vlmax), (uint16_t)vo.C, vlmax);
    for (r = 0; r < vo.H; ++r) {
        const uint16_t *px = row_ptr(x, &vi, 0, r);
        uint16_t *py = y + (size_t)r * vo.W * vo.C;
        size_t j = 0;
        while (j < (size_t)vo.W) {
            size_t q = (size_t)vo.W - j < P ? (size_t)vo.W - j : P, vl = q * vo.C;
            vuint16m8_t s = __riscv_vcompress_vm_u16m8(__riscv_vle16_v_u16m8(px + j * AG_BLOCK, q * AG_BLOCK), keep, q * AG_BLOCK);
            vuint32m4_t u = __riscv_vzext_vf2_u32m4(__riscv_vlmul_trunc_v_u16m8_u16m2(s), vl);
            vfloat32m4_t v = tanh4(__riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(u, 16, vl)), vl);
            __riscv_vse16_v_u16m2(py + j * vo.C, bf16_4(v, vl), vl);
            j += q;
        }
    }
    return 0;
}

/* ---- LADE ----
 * statistics: per channel over H x W, two-pass; the strip accumulator holds
 * 4 pixels x 32 lanes and is folded to 32 lanes after each row, row sums are
 * added to the channel totals (fp32 like the scalar kernel, which sums per
 * row too; only the within-row order differs). */
static void stats_rvv(const ag_view *v, uint16_t *p, float *mean, float *var) {
    float total[AG_BLOCK], sq[AG_BLOCK], fold[4 * AG_BLOCK];
    int32_t cb, r, j, k;
    size_t n = (size_t)v->W * AG_BLOCK, vlmax = __riscv_vsetvl_e32m4(4 * AG_BLOCK);
    int32_t groups = (int32_t)(vlmax / AG_BLOCK);      /* pixels per strip: 4 at VLEN=1024 */
    for (cb = 0; cb < v->blocks; ++cb) {
        for (j = 0; j < AG_BLOCK; ++j) total[j] = sq[j] = 0.0f;
        for (r = 0; r < v->H; ++r) {
            const uint16_t *px = row_ptr(p, v, cb, r);
            size_t i = 0, vl = vlmax;
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            while (i < n) { vl = __riscv_vsetvl_e16m2(n - i); acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, load4(px + i, vl), vl); i += vl; }
            __riscv_vse32_v_f32m4(fold, acc, vlmax);
            for (j = 0; j < AG_BLOCK; ++j) { float s = 0.0f; for (k = 0; k < groups; ++k) s += fold[k * AG_BLOCK + j]; total[j] += s; }
        }
        for (j = 0; j < AG_BLOCK; ++j) mean[cb * AG_BLOCK + j] = total[j] / (float)((int64_t)v->H * v->W);
        for (r = 0; r < v->H; ++r) {
            const uint16_t *px = row_ptr(p, v, cb, r);
            size_t i = 0, vl = vlmax;
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl), m;
            for (k = 0; k < groups; ++k) for (j = 0; j < AG_BLOCK; ++j) fold[k * AG_BLOCK + j] = mean[cb * AG_BLOCK + j];
            m = __riscv_vle32_v_f32m4(fold, vl);
            while (i < n) {
                vfloat32m4_t d;
                vl = __riscv_vsetvl_e16m2(n - i);
                d = __riscv_vfsub_vv_f32m4(load4(px + i, vl), m, vl);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, d, d, vl);
                i += vl;
            }
            __riscv_vse32_v_f32m4(fold, acc, vlmax);
            for (j = 0; j < AG_BLOCK; ++j) { float s = 0.0f; for (k = 0; k < groups; ++k) s += fold[k * AG_BLOCK + j]; sq[j] += s; }
        }
        for (j = 0; j < AG_BLOCK; ++j) var[cb * AG_BLOCK + j] = sq[j] / (float)((int64_t)v->H * v->W);
    }
}

static int lade_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]) {
    ag_view vx, vt, vo;
    uint16_t *x, *t, *y;
    float mean_x[4 * AG_BLOCK], var_x[4 * AG_BLOCK], mean_t[4 * AG_BLOCK], var_t[4 * AG_BLOCK], scale[4 * AG_BLOCK], lane[3][4 * AG_BLOCK];
    float eps = node->options.floats[0];
    int32_t cb, r, j, k;
    if (node->input_count != 2) return AG_AME_UNSUPPORTED;
    if (act(model, inputs[0], tensor_data, &vx, &x) || act(model, inputs[1], tensor_data, &vt, &t) || act(model, outputs[0], tensor_data, &vo, &y)) return AG_AME_UNSUPPORTED;
    if (!same_geometry(&vx, &vo) || !same_geometry(&vt, &vo) || vo.blocks > 4) return AG_AME_UNSUPPORTED;
    if (__riscv_vsetvl_e32m4(4 * AG_BLOCK) % AG_BLOCK || (size_t)vo.W % (__riscv_vsetvl_e32m4(4 * AG_BLOCK) / AG_BLOCK)) return AG_AME_UNSUPPORTED;
    stats_rvv(&vx, x, mean_x, var_x);
    stats_rvv(&vt, t, mean_t, var_t);
    for (j = 0; j < vo.blocks * AG_BLOCK; ++j) scale[j] = j < vo.C ? __builtin_sqrtf(var_t[j] + eps) / __builtin_sqrtf(var_x[j] + eps) : 0.0f;
    for (cb = 0; cb < vo.blocks; ++cb) {
        size_t vl = __riscv_vsetvl_e32m4(4 * AG_BLOCK), n = (size_t)vo.W * AG_BLOCK;
        vfloat32m4_t vm, vs, vmt;
        for (k = 0; k < (int32_t)(vl / AG_BLOCK); ++k) for (j = 0; j < AG_BLOCK; ++j) {
            int32_t c = cb * AG_BLOCK + j;
            lane[0][k * AG_BLOCK + j] = c < vo.C ? mean_x[c] : 0.0f;
            lane[1][k * AG_BLOCK + j] = scale[c];
            lane[2][k * AG_BLOCK + j] = c < vo.C ? mean_t[c] : 0.0f;
        }
        vm = __riscv_vle32_v_f32m4(lane[0], vl); vs = __riscv_vle32_v_f32m4(lane[1], vl); vmt = __riscv_vle32_v_f32m4(lane[2], vl);
        for (r = 0; r < vo.H; ++r) {
            const uint16_t *px = row_ptr(x, &vx, cb, r);
            uint16_t *py = row_ptr(y, &vo, cb, r);
            size_t i = 0;
            while (i < n) {
                vfloat32m4_t v;
                vl = __riscv_vsetvl_e16m2(n - i);
                v = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(__riscv_vfsub_vv_f32m4(load4(px + i, vl), vm, vl), vs, vl), vmt, vl);
                store4(py + i, leaky4(v, node->options.integers[0], node->options.floats[1], vl), vl);
                i += vl;
            }
        }
    }
    return 0;
}

/* ---- MIRROR_PAD ----
 * Copy variant (input and output are different buffers): rows are copied
 * with memcpy (RVV memcpy under AG_RVV) from the reflected source row, the
 * left / right border pixels are 64 B copies. In-place variant (the
 * output aliases the input's padded buffer): only the border is written.
 * Dense (graph input, C <= 32) source: each output row's interior is the
 * reflected dense row expanded to one channel block, unit-stride only: a
 * strip of P x C dense values is loaded (P = 16 pixels at e16 LMUL=8,
 * VLEN=1024), vrgather places value p*C+c at lane p*32+c (index >= VLMAX
 * gives the zero lanes c >= C) and one store writes P x 32 lanes. */
static inline int32_t reflect(int32_t i, int32_t n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return i; }

static void pad_border(uint16_t *dst, int32_t left, int32_t right, int32_t W, int32_t OW) {
    int32_t xx;
    for (xx = 0; xx < left; ++xx) memcpy(dst + (size_t)xx * AG_BLOCK, dst + (size_t)(left + reflect(xx - left, W)) * AG_BLOCK, AG_BLOCK * 2);
    for (xx = OW - right; xx < OW; ++xx) memcpy(dst + (size_t)xx * AG_BLOCK, dst + (size_t)(left + reflect(xx - left, W)) * AG_BLOCK, AG_BLOCK * 2);
}

static int pad_dense_rvv(const ag_view *vi, const uint16_t *x, const ag_view *vo, uint16_t *y, int32_t top, int32_t left, int32_t right) {
    size_t vlmax = __riscv_vsetvlmax_e16m8(), P = vlmax / AG_BLOCK;
    vuint16m8_t j, lane, idx;
    int32_t yy;
    if (vi->C > AG_BLOCK || !P) return AG_AME_UNSUPPORTED;
    j = __riscv_vid_v_u16m8(vlmax);
    lane = __riscv_vand_vx_u16m8(j, AG_BLOCK - 1, vlmax);
    idx = __riscv_vadd_vv_u16m8(__riscv_vmul_vx_u16m8(__riscv_vsrl_vx_u16m8(j, 5, vlmax), (uint16_t)vi->C, vlmax), lane, vlmax);
    idx = __riscv_vmerge_vxm_u16m8(idx, 0xFFFFu, __riscv_vmsgeu_vx_u16m8_b2(lane, (uint16_t)vi->C, vlmax), vlmax);
    for (yy = 0; yy < vo->H; ++yy) {
        uint16_t *dst = row_ptr(y, vo, 0, yy);
        const uint16_t *src = x + (size_t)reflect(yy - top, vi->H) * vi->W * vi->C;
        size_t p = 0;
        while (p < (size_t)vi->W) {
            size_t q = (size_t)vi->W - p < P ? (size_t)vi->W - p : P;
            vuint16m8_t s = __riscv_vle16_v_u16m8(src + p * vi->C, q * vi->C);
            __riscv_vse16_v_u16m8(dst + (left + p) * AG_BLOCK, __riscv_vrgather_vv_u16m8(s, idx, q * AG_BLOCK), q * AG_BLOCK);
            p += q;
        }
        pad_border(dst, left, right, vi->W, vo->W);
    }
    return 0;
}

static int pad_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t top = node->options.integers[1], bottom = node->options.integers[2], left = node->options.integers[3], right = node->options.integers[4];
    int32_t cb, yy;
    int in_place;
    if (node->options.integers[0] != 0) return AG_AME_UNSUPPORTED;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return AG_AME_UNSUPPORTED;
    if (!vo.block || vo.C != vi.C || vo.H != vi.H + top + bottom || vo.W != vi.W + left + right) return AG_AME_UNSUPPORTED;
    in_place = (model->tensors[outputs[0]].flags & 8) != 0;
    if (in_place && x != y) return AG_AME_UNSUPPORTED;
    if (!vi.block) return in_place ? AG_AME_UNSUPPORTED : pad_dense_rvv(&vi, x, &vo, y, top, left, right);
    for (cb = 0; cb < vo.blocks; ++cb) {
        for (yy = 0; yy < vo.H; ++yy) {
            int32_t sy = reflect(yy - top, vi.H), interior = yy >= top && yy < top + vi.H;
            uint16_t *dst = row_ptr(y, &vo, cb, yy);
            const uint16_t *src = row_ptr(x, &vi, cb, sy);
            if (!(in_place && interior))   /* interior rows of an in-place pad already hold the input */
                memcpy(dst + (size_t)left * AG_BLOCK, src, (size_t)vi.W * AG_BLOCK * 2);
            pad_border(dst, left, right, vi.W, vo.W);
        }
    }
    return 0;
}

int ag_execute_node_rvv(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]) {
    switch (node->opcode) {
    case 0: return elementwise_rvv(model, node, inputs, outputs, tensor_data, 1);
    case 98: return elementwise_rvv(model, node, inputs, outputs, tensor_data, 0);
    case 23: return resize_rvv(model, node, inputs, outputs, tensor_data);
    case 28: return tanh_rvv(model, node, inputs, outputs, tensor_data);
    case 200: return lade_rvv(model, node, inputs, outputs, tensor_data);
    case 100: return pad_rvv(model, node, inputs, outputs, tensor_data);
    default: return AG_AME_UNSUPPORTED;
    }
}

/* ---- convolution epilogue: fp32 C rows [rows][KO] -> blocked bf16 output pixels ---- */
void ag_conv_epilogue_rvv(uint16_t *y_row0, size_t plane_elems, const float *cbuf, int32_t rows, int32_t KO, int32_t blocks,
                          const float *bias, int fused, float alpha) {
    int32_t cb, r;
    for (cb = 0; cb < blocks; ++cb) {
        int32_t lanes = KO - cb * AG_BLOCK < AG_BLOCK ? KO - cb * AG_BLOCK : AG_BLOCK;
        size_t vl = __riscv_vsetvl_e32m1(AG_BLOCK);
        vfloat32m1_t vb = bias ? __riscv_vle32_v_f32m1(bias + cb * AG_BLOCK, (size_t)lanes) : __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vbool32_t valid = __riscv_vmsltu_vx_u32m1_b32(__riscv_vid_v_u32m1(vl), (uint32_t)lanes, vl);
        if (bias && lanes < AG_BLOCK) vb = __riscv_vfmerge_vfm_f32m1(vb, 0.0f, __riscv_vmnot_m_b32(valid, vl), vl);
        for (r = 0; r < rows; ++r) {
            vfloat32m1_t v = __riscv_vle32_v_f32m1(cbuf + (size_t)r * KO + cb * AG_BLOCK, (size_t)lanes);
            v = __riscv_vfadd_vv_f32m1(v, vb, vl);
            if (fused == AG_FUSED_LEAKY) {
                vbool32_t neg = __riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl);
                v = __riscv_vmerge_vvm_f32m1(v, __riscv_vfmul_vf_f32m1(v, alpha, vl), neg, vl);
            }
            v = __riscv_vfmerge_vfm_f32m1(v, 0.0f, __riscv_vmnot_m_b32(valid, vl), vl);
            store1(y_row0 + (size_t)cb * plane_elems + (size_t)r * AG_BLOCK, v, vl);
        }
    }
}

/* ---- board frame conversions: UINT8 RGB <-> dense bf16 ---- */
void ag_rvv_u8_to_bf16(uint16_t *dst, const uint8_t *src, size_t n) {   /* x / 127.5 - 1 */
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(__riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(src + i, vl), vl), vl);
        store4(dst + i, __riscv_vfsub_vf_f32m4(__riscv_vfdiv_vf_f32m4(f, 127.5f, vl), 1.0f, vl), vl);
        i += vl;
    }
}

void ag_rvv_bf16_to_u8(uint8_t *dst, const uint16_t *src, size_t n) {   /* clip((y + 1) * 127.5, 0, 255) truncated */
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m2(n - i);
        vfloat32m4_t f = __riscv_vfmul_vf_f32m4(__riscv_vfadd_vf_f32m4(load4(src + i, vl), 1.0f, vl), 127.5f, vl);
        vuint32m4_t u = __riscv_vfcvt_rtz_xu_f_v_u32m4(__riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(f, 0.0f, vl), 255.0f, vl), vl);
        __riscv_vse8_v_u8m1(dst + i, __riscv_vnclipu_wx_u8m1(__riscv_vnclipu_wx_u16m2(u, 0, __RISCV_VXRM_RDN, vl), 0, __RISCV_VXRM_RDN, vl), vl);
        i += vl;
    }
}
