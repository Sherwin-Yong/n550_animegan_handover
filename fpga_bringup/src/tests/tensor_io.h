#ifndef ANIMEGAN_TESTS_TENSOR_IO_H
#define ANIMEGAN_TESTS_TENSOR_IO_H
#include <animegan/bf16.h>
#include <animegan/platform.h>
#include <animegan/runtime.h>

/* Conversions between the reference files (fp32, logical NHWC, no padding)
 * and the runtime storage (bf16 blocked / dense) plus float difference
 * statistics, shared by the node and forward test drivers. Numbers are
 * printed as integers in micro units (no float formatting in the platform
 * layer). */

/* fp32 NHWC -> storage of `desc` (blocked bf16 with zero lanes beyond C, or
 * dense bf16; a lane-packed tensor also gets its neighbourhood lanes) */
static inline void ag_to_storage(const ag_tensor_desc *desc, const float *src, void *dst) {
    ag_view v = ag_view_of(desc);
    uint16_t *p = dst;
    int32_t y, x, c;
    for (y = 0; y < v.H; ++y)
        for (x = 0; x < v.W; ++x)
            for (c = 0; c < (v.block ? v.blocks * AG_BLOCK : v.C); ++c)
                p[ag_index(&v, y, x, c)] = c < v.C ? ag_f32_to_bf16(src[((size_t)y * v.W + x) * v.C + c]) : 0;
    if (desc->flags & AG_TENSOR_LANE_PACK) ag_fill_lane_pack(desc, p);
}

/* storage of `desc` -> fp32 NHWC (logical elements only) */
static inline void ag_from_storage(const ag_tensor_desc *desc, const void *src, float *dst) {
    ag_view v = ag_view_of(desc);
    const uint16_t *p = src;
    int32_t y, x, c;
    for (y = 0; y < v.H; ++y)
        for (x = 0; x < v.W; ++x)
            for (c = 0; c < v.C; ++c)
                dst[((size_t)y * v.W + x) * v.C + c] = ag_bf16_to_f32(p[ag_index(&v, y, x, c)]);
}

typedef struct {
    size_t count, diff;
    float max_abs, max_ref, sum_abs;
    size_t first;
} ag_fdiff;

static inline void ag_compare_f32(const float *got, const float *want, size_t count, ag_fdiff *st) {
    size_t i;
    st->count = count; st->diff = 0; st->max_abs = 0.0f; st->max_ref = 0.0f; st->sum_abs = 0.0f; st->first = 0;
    for (i = 0; i < count; ++i) {
        float d = got[i] - want[i], a = want[i] < 0.0f ? -want[i] : want[i];
        if (d < 0.0f) d = -d;
        if (a > st->max_ref) st->max_ref = a;
        if (d == 0.0f) continue;
        if (!st->diff) st->first = i;
        ++st->diff;
        st->sum_abs += d;
        if (d > st->max_abs) st->max_abs = d;
    }
}

static inline void ag_put_micro(float v) { ag_plat_put_int((long)(v * 1000000.0f + (v >= 0.0f ? 0.5f : -0.5f))); }

/* "diff=<n>/<count> max_abs_u=<micro> max_ref_u=<micro> mean_abs_u=<micro> rel_u=<micro>" */
static inline void ag_print_fdiff(const ag_fdiff *st) {
    float rel = st->max_ref > 0.0f ? st->max_abs / st->max_ref : st->max_abs;
    ag_plat_puts("diff="); ag_plat_put_int((long)st->diff); ag_plat_puts("/"); ag_plat_put_int((long)st->count);
    ag_plat_puts(" max_abs_u="); ag_put_micro(st->max_abs);
    ag_plat_puts(" max_ref_u="); ag_put_micro(st->max_ref);
    ag_plat_puts(" mean_abs_u="); ag_put_micro(st->count ? st->sum_abs / (float)st->count : 0.0f);
    ag_plat_puts(" rel_u="); ag_put_micro(rel);
}

/* "stats node <id> sum_u <micro> abs_u <micro> max_u <micro>" (ag_tensor_stats of
 * the node's output; board second forward and forward --stats, tools/check.py boardlog) */
static inline void ag_print_stats(uint32_t node_id, const ag_stats *st) {
    ag_plat_puts("stats node "); ag_plat_put_int((long)node_id);
    ag_plat_puts(" sum_u "); ag_plat_put_int((long)(st->sum * 1e6 + (st->sum >= 0.0 ? 0.5 : -0.5)));
    ag_plat_puts(" abs_u "); ag_plat_put_int((long)(st->abs_sum * 1e6 + 0.5));
    ag_plat_puts(" max_u "); ag_put_micro(st->max_abs);
    ag_plat_puts("\n");
}

/* tolerance code: 0 rel <= 1%, 12 rel <= 5%, 11 beyond (rel = max|diff| / max|ref|) */
static inline int ag_fdiff_code(const ag_fdiff *st) {
    float rel = st->max_ref > 0.0f ? st->max_abs / st->max_ref : st->max_abs;
    return rel <= 0.01f ? 0 : rel <= 0.05f ? 12 : 11;
}

#endif
