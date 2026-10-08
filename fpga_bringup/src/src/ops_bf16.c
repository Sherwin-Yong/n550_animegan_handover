#include <animegan/bf16.h>
#include <animegan/runtime.h>

/* Scalar bf16 kernels (reference semantics for every build): MIRROR_PAD, ADD,
 * RESIZE_BILINEAR, LEAKY_RELU, TANH, LADE. Elementwise math is fp32, results
 * are rounded to bf16 on store. Blocked tensors keep the lanes beyond C at
 * zero so a following convolution can read whole channel blocks. */

static int act(const ag_model *model, int32_t id, void *const tensor_data[], ag_view *v, uint16_t **p) {
    const ag_tensor_desc *t;
    if (id < 0 || (uint32_t)id >= model->tensor_count || !tensor_data[id]) return -1;
    t = model->tensors + id;
    if (t->rank != 4 || t->type != AG_TYPE_BF16) return -2;
    *v = ag_view_of(t);
    *p = tensor_data[id];
    return 0;
}

static inline float leaky(float v, int fused, float alpha) {
    return (fused == AG_FUSED_LEAKY && v < 0.0f) ? v * alpha : v;
}

static inline int32_t reflect(int32_t i, int32_t n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i;
}

#ifndef AG_RVV
void ag_tensor_stats(const ag_tensor_desc *tensor, const void *data, ag_stats *st) {
    ag_view v = ag_view_of(tensor);
    const uint16_t *p = data;
    int32_t y, x, c;
    st->sum = st->abs_sum = 0.0;
    st->max_abs = 0.0f;
    for (y = 0; y < v.H; ++y)
        for (x = 0; x < v.W; ++x)
            for (c = 0; c < v.C; ++c) {
                float f = ag_bf16_to_f32(p[ag_index(&v, y, x, c)]), a = f < 0.0f ? -f : f;
                st->sum += f;
                st->abs_sum += a;
                if (a > st->max_abs) st->max_abs = a;
            }
}
#endif

/* Lanes beyond C of a lane-packed tensor (AG_TENSOR_LANE_PACK, runtime.h),
 * lanes from R*KW*C on are zero: only lanes < C are read, so the fill can run
 * in place in any order. */
void ag_fill_lane_pack(const ag_tensor_desc *tensor, uint16_t *p) {
    ag_view v = ag_view_of(tensor);
    int32_t R = (int32_t)(tensor->reserved & 0xFFu), KW = (int32_t)(tensor->reserved >> 8 & 0xFFu), y, x, lane;
    if (!v.block || v.C < 1 || v.C > AG_BLOCK || R < 1 || KW < 1) return;
    for (y = 0; y < v.H; ++y)
        for (x = 0; x < v.W; ++x)
            for (lane = v.C; lane < AG_BLOCK; ++lane) {
                int32_t k = lane / v.C, c = lane % v.C, dy = k / KW, dx = k % KW;
                p[ag_index(&v, y, x, lane)] = dy < R && y + dy < v.H && x + dx < v.W ? p[ag_index(&v, y + dy, x + dx, c)] : 0;
            }
}

/* The output either is its own buffer (copy) or aliases the input's padded
 * buffer (flag 8): then the interior already holds the input and only the
 * reflected border is written. A lane-packed output gets its neighbourhood
 * lanes afterwards. */
int ag_mirror_pad(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                  void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t top = node->options.integers[1], left = node->options.integers[3], yy, xx, c;
    int in_place;
    if (node->input_count < 1 || node->options.integers[0] != 0) return -30;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (!vo.block || vo.C != vi.C || vo.H != vi.H + top + node->options.integers[2] || vo.W != vi.W + left + node->options.integers[4]) return -32;
    if (top >= vi.H || left >= vi.W) return -33;
    in_place = (model->tensors[outputs[0]].flags & 8) != 0;
    if (in_place && x != y) return -34;
    for (yy = 0; yy < vo.H; ++yy) {
        int32_t sy = reflect(yy - top, vi.H);
        int32_t interior_row = yy >= top && yy < top + vi.H;
        for (xx = 0; xx < vo.W; ++xx) {
            int32_t sx = reflect(xx - left, vi.W);
            if (in_place && interior_row && xx >= left && xx < left + vi.W) continue;   /* already in place */
            for (c = 0; c < vo.blocks * AG_BLOCK; ++c)
                y[ag_index(&vo, yy, xx, c)] = c < vi.C ? x[ag_index(&vi, sy, sx, c)] : 0;
        }
    }
    if (model->tensors[outputs[0]].flags & AG_TENSOR_LANE_PACK) ag_fill_lane_pack(model->tensors + outputs[0], y);
    return 0;
}

int ag_add(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
           void *const tensor_data[]) {
    ag_view va, vb, vo;
    uint16_t *a, *b, *y;
    int32_t yy, xx, c;
    if (node->input_count != 2) return -30;
    if (act(model, inputs[0], tensor_data, &va, &a) || act(model, inputs[1], tensor_data, &vb, &b) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (!vo.block || va.H != vo.H || va.W != vo.W || va.C != vo.C || vb.H != vo.H || vb.W != vo.W || vb.C != vo.C) return -32;
    for (yy = 0; yy < vo.H; ++yy)
        for (xx = 0; xx < vo.W; ++xx)
            for (c = 0; c < vo.blocks * AG_BLOCK; ++c) {
                float v = c < vo.C ? ag_bf16_to_f32(a[ag_index(&va, yy, xx, c)]) + ag_bf16_to_f32(b[ag_index(&vb, yy, xx, c)]) : 0.0f;
                y[ag_index(&vo, yy, xx, c)] = ag_f32_to_bf16(leaky(v, node->options.integers[0], node->options.floats[0]));
            }
    return 0;
}

int ag_leaky_relu(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                  void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t yy, xx, c;
    float alpha = node->options.floats[0];
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (!vo.block || vi.H != vo.H || vi.W != vo.W || vi.C != vo.C) return -32;
    for (yy = 0; yy < vo.H; ++yy)
        for (xx = 0; xx < vo.W; ++xx)
            for (c = 0; c < vo.blocks * AG_BLOCK; ++c) {
                float v = c < vo.C ? leaky(ag_bf16_to_f32(x[ag_index(&vi, yy, xx, c)]), AG_FUSED_LEAKY, alpha) : 0.0f;
                y[ag_index(&vo, yy, xx, c)] = ag_f32_to_bf16(v);
            }
    return 0;
}

/* tanh without libm: 1 - 2 / (exp(2x) + 1) with exp by range reduction to
 * [-ln2/2, ln2/2] and a degree-6 polynomial (relative error ~1e-7, far below
 * bf16 resolution). */
static float exp_f32(float x) {
    const float ln2 = 0.69314718f, inv_ln2 = 1.44269504f;
    float n = (float)(int)(x * inv_ln2 + (x >= 0.0f ? 0.5f : -0.5f));
    float f = x - n * ln2, p, r;
    ag_f32_bits s;
    p = 1.0f / 720.0f;
    p = p * f + 1.0f / 120.0f;
    p = p * f + 1.0f / 24.0f;
    p = p * f + 1.0f / 6.0f;
    p = p * f + 0.5f;
    p = p * f + 1.0f;
    r = p * f + 1.0f;
    s.u = (uint32_t)((int)n + 127) << 23;
    return r * s.f;
}

float ag_tanhf(float x) {
    if (x > 9.0f) return 1.0f;
    if (x < -9.0f) return -1.0f;
    return 1.0f - 2.0f / (exp_f32(2.0f * x) + 1.0f);
}

/* TANH is the last node: blocked input, dense NHWC graph output */
int ag_tanh(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
            void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t yy, xx, c;
    (void)node;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (vi.H != vo.H || vi.W != vo.W || vi.C != vo.C) return -32;
    for (yy = 0; yy < vo.H; ++yy)
        for (xx = 0; xx < vo.W; ++xx)
            for (c = 0; c < vo.blocks * AG_BLOCK && (vo.block || c < vo.C); ++c) {
                float v = c < vo.C ? ag_tanhf(ag_bf16_to_f32(x[ag_index(&vi, yy, xx, c)])) : 0.0f;
                y[ag_index(&vo, yy, xx, c)] = ag_f32_to_bf16(v);
            }
    return 0;
}

/* x2 bilinear upsample, asymmetric coordinates (tf1 resize_images with
 * align_corners=False): src = dst * (in / out); the top-left sample and its
 * clamped right / bottom neighbours are averaged with the fractional part. */
int ag_resize_bilinear(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                       void *const tensor_data[]) {
    ag_view vi, vo;
    uint16_t *x, *y;
    int32_t yy, xx, c;
    float sy = 0.0f, sx = 0.0f;
    if (node->options.integers[0] || node->options.integers[1]) return -30;
    if (act(model, inputs[0], tensor_data, &vi, &x) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (!vo.block || vi.C != vo.C || vo.H < vi.H || vo.W < vi.W) return -32;
    sy = (float)vi.H / (float)vo.H; sx = (float)vi.W / (float)vo.W;
    for (yy = 0; yy < vo.H; ++yy) {
        float fy = (float)yy * sy;
        int32_t y0 = (int32_t)fy, y1 = y0 + 1 < vi.H ? y0 + 1 : vi.H - 1;
        float wy = fy - (float)y0;
        for (xx = 0; xx < vo.W; ++xx) {
            float fx = (float)xx * sx;
            int32_t x0 = (int32_t)fx, x1 = x0 + 1 < vi.W ? x0 + 1 : vi.W - 1;
            float wx = fx - (float)x0;
            for (c = 0; c < vo.blocks * AG_BLOCK; ++c) {
                float v = 0.0f;
                if (c < vo.C) {
                    float a = ag_bf16_to_f32(x[ag_index(&vi, y0, x0, c)]), b = ag_bf16_to_f32(x[ag_index(&vi, y0, x1, c)]);
                    float d = ag_bf16_to_f32(x[ag_index(&vi, y1, x0, c)]), e = ag_bf16_to_f32(x[ag_index(&vi, y1, x1, c)]);
                    float top = a + (b - a) * wx, bottom = d + (e - d) * wx;
                    v = top + (bottom - top) * wy;
                }
                y[ag_index(&vo, yy, xx, c)] = ag_f32_to_bf16(v);
            }
        }
    }
    return 0;
}

/* LADE: out = (x - mean_x) / sqrt(var_x + eps) * sqrt(var_tx + eps) + mean_tx
 * per channel over H x W, optionally followed by LeakyReLU. Statistics are
 * two-pass (mean, then mean of squared deviation) in fp32 with per-row
 * partial sums. Lanes beyond C are zero in both inputs and stay zero. */
static float sqrt_f32(float v) { return __builtin_sqrtf(v); }

static void channel_stats(const ag_view *v, const uint16_t *p, float *mean, float *var) {
    int32_t c, yy, xx;
    for (c = 0; c < v->C; ++c) {
        float total = 0.0f, sq = 0.0f, m;
        for (yy = 0; yy < v->H; ++yy) {
            float row = 0.0f;
            for (xx = 0; xx < v->W; ++xx) row += ag_bf16_to_f32(p[ag_index(v, yy, xx, c)]);
            total += row;
        }
        m = total / (float)((int64_t)v->H * v->W);
        for (yy = 0; yy < v->H; ++yy) {
            float row = 0.0f;
            for (xx = 0; xx < v->W; ++xx) {
                float d = ag_bf16_to_f32(p[ag_index(v, yy, xx, c)]) - m;
                row += d * d;
            }
            sq += row;
        }
        mean[c] = m;
        var[c] = sq / (float)((int64_t)v->H * v->W);
    }
}

int ag_lade(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
            void *const tensor_data[]) {
    ag_view vx, vt, vo;
    uint16_t *x, *t, *y;
    float mean_x[AG_BLOCK * 4], var_x[AG_BLOCK * 4], mean_t[AG_BLOCK * 4], var_t[AG_BLOCK * 4], scale[AG_BLOCK * 4];
    float eps = node->options.floats[0];
    int32_t yy, xx, c;
    if (node->input_count != 2) return -30;
    if (act(model, inputs[0], tensor_data, &vx, &x) || act(model, inputs[1], tensor_data, &vt, &t) || act(model, outputs[0], tensor_data, &vo, &y)) return -31;
    if (!vo.block || vx.H != vo.H || vx.W != vo.W || vx.C != vo.C || vt.H != vo.H || vt.W != vo.W || vt.C != vo.C) return -32;
    if (vo.C > AG_BLOCK * 4) return -33;
    channel_stats(&vx, x, mean_x, var_x);
    channel_stats(&vt, t, mean_t, var_t);
    for (c = 0; c < vo.C; ++c) scale[c] = sqrt_f32(var_t[c] + eps) / sqrt_f32(var_x[c] + eps);
    for (yy = 0; yy < vo.H; ++yy)
        for (xx = 0; xx < vo.W; ++xx)
            for (c = 0; c < vo.blocks * AG_BLOCK; ++c) {
                float v = 0.0f;
                if (c < vo.C) v = (ag_bf16_to_f32(x[ag_index(&vx, yy, xx, c)]) - mean_x[c]) * scale[c] + mean_t[c];
                y[ag_index(&vo, yy, xx, c)] = ag_f32_to_bf16(leaky(v, node->options.integers[0], node->options.floats[1]));
            }
    return 0;
}
