#ifndef ANIMEGAN_RUNTIME_H
#define ANIMEGAN_RUNTIME_H
#include <stddef.h>
#include <stdint.h>

/* bf16 runtime for the Ghibli-c1 graph. Activations are bf16 in a
 * channel-blocked layout [C/32][H][W][32] (one pixel of a channel block = 64
 * bytes, so 128 consecutive pixels are one contiguous AME tile); conv weights
 * are bf16 tile-major [(ky,kx,cb)][KO][32]; biases and the LADE statistics are
 * fp32. The graph input and output are dense NHWC bf16 [1][512][512][3]. */
#define AG_MAX_RANK 4u
#define AG_OPTION_INTS 8u
#define AG_OPTION_FLOATS 2u
#define AG_PACK_VERSION 3u
#define AG_TYPE_F32 0u
#define AG_TYPE_I32 2u
#define AG_TYPE_BF16 18u
#define AG_BLOCK 32
#define AG_FUSED_LEAKY 6     /* fusedActivationFunction value: LeakyReLU(alpha = floats[0]) */

typedef enum {
    AG_BACKEND_SCALAR = 0,
    AG_BACKEND_AME = 1
} ag_backend;

/* flags: 1 constant, 2 graph input, 4 graph output, 8 alias of the node's first input (padded buffer),
 * AG_TENSOR_LANE_PACK: blocked tensor (C <= 32, no pad border) whose lanes beyond C carry the
 * neighbourhood for a K-packed convolution: lane (dy*KW + dx)*C + c of pixel (y, x) = lane c of pixel
 * (y+dy, x+dx) for dy < R, dx < KW, 0 outside the H x W image; reserved = R | KW << 8. The conv
 * reading it has options.integers[6] = id + 1 of its repacked weights (tools/model_pack.py). */
#define AG_TENSOR_LANE_PACK (1u << 16)
typedef struct {
    uint32_t id;
    uint32_t type;
    uint32_t rank;
    uint32_t flags;
    int32_t shape[AG_MAX_RANK];  /* activations [1,H,W,C]; conv weights [KO,KH,KW,C]; bias [KO] */
    int32_t block;               /* AG_BLOCK: channel-blocked storage; 0: dense row-major */
    int32_t pad_h;               /* blocked activations: border rows / columns allocated around the */
    int32_t pad_w;               /* logical image (0 = none) */
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t reserved2;
    uint64_t reserved3;
} ag_tensor_desc;

typedef struct {
    int32_t integers[AG_OPTION_INTS];
    float floats[AG_OPTION_FLOATS];
} ag_op_options;

/* opcodes: 100 MIRROR_PAD (ints: mode, top, bottom, left, right), 3 CONV_2D (ints[1..2] strides,
 * ints[5] fused activation, floats[0] alpha), 0 ADD (ints[0] fused, floats[0] alpha),
 * 23 RESIZE_BILINEAR (x2, asymmetric), 98 LEAKY_RELU (floats[0] alpha), 28 TANH,
 * 200 LADE (inputs x, tx; floats[0] eps; ints[0] fused, floats[1] alpha) */
typedef struct {
    uint32_t id;
    uint32_t opcode;
    uint32_t version;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t input_offset;
    uint64_t output_offset;
    ag_op_options options;
} ag_node_desc;

typedef struct {
    uint8_t magic[8];
    uint32_t version;
    uint32_t endian_tag;
    uint32_t tensor_count;
    uint32_t node_count;
    uint64_t tensor_offset;
    uint64_t node_offset;
    uint64_t total_size;
    uint64_t reserved;
} ag_pack_header;

typedef struct {
    const uint8_t *data;
    size_t size;
    const ag_tensor_desc *tensors;
    const ag_node_desc *nodes;
    uint32_t tensor_count;
    uint32_t node_count;
} ag_model;

typedef struct {
    const ag_model *model;
    void **tensor_data;
    size_t *tensor_size;
    uint32_t capacity;
    ag_backend backend;
    void *scratch;        /* operator workspace (ag_node_scratch_bytes), may be null */
    size_t scratch_size;
} ag_runtime;

/* Geometry of an activation buffer: logical H x W x C, storage pitch (pixels
 * per padded row), plane (pixels per channel block) and origin (pixel index of
 * logical (0,0)). block == 0 is the dense NHWC graph input / output. */
typedef struct {
    int32_t H, W, C, blocks, block;
    int32_t pitch, plane, origin;
} ag_view;
ag_view ag_view_of(const ag_tensor_desc *tensor);
static inline size_t ag_index(const ag_view *v, int32_t y, int32_t x, int32_t c) {
    if (v->block) return ((size_t)(c / AG_BLOCK) * (size_t)v->plane + (size_t)v->origin + (size_t)y * (size_t)v->pitch + (size_t)x) * AG_BLOCK + (size_t)(c % AG_BLOCK);
    return ((size_t)y * (size_t)v->W + (size_t)x) * (size_t)v->C + (size_t)c;
}

/* Static activation plan: 64-byte aligned arena offsets per runtime tensor
 * (UINT64_MAX = not placed: constants and the graph input) plus one scratch
 * block for the largest per-node workspace of the backend. The arena ends with
 * AG_PLAN_TAIL readable bytes for the matrix unit's tile over-read. */
#define AG_PLAN_TAIL 8192u
typedef struct {
    uint64_t arena_bytes;
    uint64_t scratch_offset;
    uint64_t scratch_bytes;
    uint32_t placed;
} ag_plan;
int ag_plan_build(const ag_model *model, ag_backend backend, uint64_t offsets[], ag_plan *plan);
uint32_t ag_plan_check(const ag_model *model, const uint64_t offsets[], const ag_plan *plan);
size_t ag_node_scratch_bytes(const ag_model *model, uint32_t node_id, ag_backend backend);

int ag_model_open(ag_model *model, const void *data, size_t size);
int ag_model_check_range(const ag_model *model, uint64_t offset, uint64_t size);
const void *ag_model_at(const ag_model *model, uint64_t offset, uint64_t size);
size_t ag_tensor_elements(const ag_tensor_desc *tensor);   /* logical element count */
size_t ag_tensor_bytes(const ag_tensor_desc *tensor);      /* storage bytes (blocked, padded) */
int ag_execute_node_scalar(const ag_model *model, uint32_t node_id, void *const tensor_data[]);

/* Statistics of the logical elements of a rank-4 bf16 tensor (blocked: lanes < C
 * of the interior; dense: all): sum and sum of |x| accumulated in double, max |x|.
 * One definition, two implementations: scalar (src/ops_bf16.c) and, in AG_RVV
 * builds, RVV (src/ops_rvv.c; the board's tensors are RVV-only). The board's
 * second forward and tests/forward --stats print them per node for
 * tools/check.py boardlog. */
typedef struct {
    double sum, abs_sum;
    float max_abs;
} ag_stats;
void ag_tensor_stats(const ag_tensor_desc *tensor, const void *data, ag_stats *st);

/* scalar kernels (reference semantics, every build) */
int ag_mirror_pad(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
void ag_fill_lane_pack(const ag_tensor_desc *tensor, uint16_t *data);   /* lanes beyond C from lanes < C (AG_TENSOR_LANE_PACK) */
int ag_conv2d_scalar(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
int ag_add(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
int ag_resize_bilinear(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
int ag_leaky_relu(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
int ag_tanh(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
int ag_lade(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
float ag_tanhf(float x);

/* conv geometry shared by the scalar and AME kernels */
typedef struct {
    ag_view in, out;
    const uint16_t *x, *w;
    const float *bias;
    uint16_t *y;
    int32_t KO, KH, KW, C, CB, stride_h, stride_w, fused;
    float alpha;
} ag_conv_geom;
int ag_conv2d_check(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                    void *const tensor_data[], ag_conv_geom *g);

/* Returned by AME / RVV kernels for shapes they do not cover; the executor
 * then runs the scalar kernel for that node. */
#define AG_AME_UNSUPPORTED 1
/* AME bf16 GEMM (RV64 only): C[M][N] fp32 = A[M][K] . B[N][K]^T, A and B bf16 with byte strides */
int ag_gemm_bf16_ame(float *c, long c_stride, const uint16_t *a, long a_stride,
                     const uint16_t *b, long b_stride, int M, int K, int N);
int ag_conv2d_ame(const ag_model *model, const ag_node_desc *node, const int32_t *inputs, const int32_t *outputs,
                  void *const tensor_data[], void *scratch, size_t scratch_size);
extern int ag_conv_overlap;   /* 1 (default): epilogue of a tile group overlaps the next group's matrix steps */
/* AME conv workspace (src/conv_ame.c checks it): two sets of 3 C blocks of up to
 * 128 x 128 fp32; ag_node_scratch_bytes uses it in every build, so the host plans
 * the board's arena too (tests/runtime_test --plan) */
#define AG_CONV_AME_SCRATCH ((size_t)2 * 3 * 128 * 128 * 4)
int ag_execute_node_rvv(const ag_model *model, const ag_node_desc *node,
                        const int32_t *inputs, const int32_t *outputs, void *const tensor_data[]);
/* RVV helpers (AG_RVV builds): convolution epilogue over fp32 C rows (state
 * prepared once per node, KO <= 128), board frame conversions */
typedef struct {
    float bias[8 * AG_BLOCK];       /* bias, zero beyond KO; KO = 32: repeated for 8 pixels */
    size_t plane_elems;
    int32_t KO, blocks, mode, wide; /* mode: LeakyReLU form; wide: KO = 32 m8 path */
    float alpha;
} ag_conv_epi;
void ag_conv_epilogue_rvv_init(ag_conv_epi *e, const ag_conv_geom *g);
void ag_conv_epilogue_rvv(const ag_conv_epi *e, uint16_t *y_row0, const float *cbuf, int32_t rows);
void ag_rvv_u8_to_bf16(uint16_t *dst, const uint8_t *src, size_t n);
void ag_rvv_bf16_to_u8(uint8_t *dst, const uint16_t *src, size_t n);

int ag_runtime_init(ag_runtime *runtime, const ag_model *model, void **tensor_data,
                    size_t *tensor_size, uint32_t capacity, ag_backend backend);
int ag_runtime_bind(ag_runtime *runtime, uint32_t tensor_id, void *data, size_t size);
int ag_runtime_execute_node(ag_runtime *runtime, uint32_t node_id);

#endif
