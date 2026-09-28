#include <animegan/bf16.h>
#include <animegan/runtime.h>

/* CONV_2D geometry check shared by the scalar and AME kernels: VALID padding
 * (the MIRROR_PAD node before it supplies the border), dilation 1, optional
 * bias, optional fused LeakyReLU. Input and output are blocked bf16, weights
 * tile-major bf16 [(ky,kx,cb)][KO][32], bias fp32 [KO]. */
int ag_conv2d_check(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                    void *const tensor_data[], ag_conv_geom *g) {
    const ag_tensor_desc *in, *wt, *out, *bs = 0;
    int32_t OH, OW;
    if (node->input_count != 3 || inputs[0] < 0 || inputs[1] < 0) return -20;
    in = model->tensors + inputs[0]; wt = model->tensors + inputs[1]; out = model->tensors + outputs[0];
    if (inputs[2] >= 0) bs = model->tensors + inputs[2];
    if (!tensor_data[inputs[0]] || !tensor_data[inputs[1]] || !tensor_data[outputs[0]] || (bs && !tensor_data[inputs[2]])) return -21;
    if (in->rank != 4 || wt->rank != 4 || out->rank != 4 || !in->block || !wt->block || !out->block) return -22;
    if (bs && (bs->type != AG_TYPE_F32 || bs->shape[0] != wt->shape[0])) return -23;
    if (node->options.integers[0] != 0 || node->options.integers[3] > 1 || node->options.integers[4] > 1) return -24;
    g->in = ag_view_of(in); g->out = ag_view_of(out);
    g->KO = wt->shape[0]; g->KH = wt->shape[1]; g->KW = wt->shape[2]; g->C = wt->shape[3];
    g->CB = (g->C + AG_BLOCK - 1) / AG_BLOCK;
    g->stride_h = node->options.integers[1] > 0 ? node->options.integers[1] : 1;
    g->stride_w = node->options.integers[2] > 0 ? node->options.integers[2] : 1;
    g->fused = node->options.integers[5];
    g->alpha = node->options.floats[0];
    if (g->C != g->in.C || g->KO != g->out.C) return -25;
    OH = (g->in.H - g->KH) / g->stride_h + 1; OW = (g->in.W - g->KW) / g->stride_w + 1;
    if (OH != g->out.H || OW != g->out.W) return -26;
    if (g->fused != 0 && g->fused != AG_FUSED_LEAKY) return -27;
    g->x = tensor_data[inputs[0]]; g->w = tensor_data[inputs[1]]; g->y = tensor_data[outputs[0]];
    g->bias = bs ? tensor_data[inputs[2]] : 0;
    return 0;
}

/* Reference kernel: fp32 accumulation over (ky, kx, channel block, lane); the
 * zero lanes beyond C in both input and weights contribute nothing. */
int ag_conv2d_scalar(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                     void *const tensor_data[]) {
    ag_conv_geom g;
    int32_t oy, ox, ko, ky, kx, cb, j;
    int rc = ag_conv2d_check(model, node, inputs, outputs, tensor_data, &g);
    if (rc) return rc;
    for (oy = 0; oy < g.out.H; ++oy) {
        for (ox = 0; ox < g.out.W; ++ox) {
            for (ko = 0; ko < g.out.blocks * AG_BLOCK; ++ko) {
                float acc;
                if (ko >= g.KO) { g.y[ag_index(&g.out, oy, ox, ko)] = 0; continue; }
                acc = g.bias ? g.bias[ko] : 0.0f;
                for (ky = 0; ky < g.KH; ++ky) {
                    for (kx = 0; kx < g.KW; ++kx) {
                        for (cb = 0; cb < g.CB; ++cb) {
                            const uint16_t *xp = g.x + ag_index(&g.in, oy * g.stride_h + ky, ox * g.stride_w + kx, cb * AG_BLOCK);
                            const uint16_t *wp = g.w + (((size_t)(ky * g.KW + kx) * g.CB + cb) * g.KO + ko) * AG_BLOCK;
                            for (j = 0; j < AG_BLOCK; ++j) acc += ag_bf16_to_f32(xp[j]) * ag_bf16_to_f32(wp[j]);
                        }
                    }
                }
                if (g.fused == AG_FUSED_LEAKY && acc < 0.0f) acc *= g.alpha;
                g.y[ag_index(&g.out, oy, ox, ko)] = ag_f32_to_bf16(acc);
            }
        }
    }
    return 0;
}
