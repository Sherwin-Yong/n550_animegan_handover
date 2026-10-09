#include <animegan/ame.h>
#include <animegan/bf16.h>
#include <animegan/board_log.h>
#include <animegan/platform.h>
#include <animegan/runtime.h>
#include <stddef.h>
#include <string.h>
#include "bsp/board.h"
#include "tensor_io.h"
#if defined(AG_AME) || defined(AG_RVV)
#include <riscv_vector.h>
#endif

/* Back-door (PCIe) frame program, one frame per boot. While the CPU is held in reset
 * the host writes
 *   the program (raw binary, model pack embedded) at BOARD_DRAM_BASE,
 *   BOARD_HOST_MAGIC at BOARD_MBOX_MAGIC (proves the PCIe writes reach DDR; the board
 *   clears it after reading, so every frame needs a fresh one),
 *   one 512x512 RGB UINT8 frame at BOARD_INPUT_ADDR;
 * then releases reset and polls the mailbox status at BOARD_MBOX_ADDR:
 *   0xB0000001 boot, 0xB0000003 running, 0xB0000004 done (result frame at
 *   BOARD_OUTPUT_ADDR), 0xB00000EE error (code at +0x04), 0xB00000FF trap
 *   (src/bsp/trap.c).
 * Around the bf16 graph the board does the model's normalisation itself:
 * pre  x = u/127.5 - 1 (UINT8 -> dense bf16 graph input),
 * post u = clip((y + 1) * 127.5, 0, 255) truncated (bf16 graph output -> UINT8),
 * with RVV. Stage cycle counts come from mcycle (BOARD_CPU_HZ = 40 MHz on the
 * board, arbitrary on QEMU). Frames, graph input and output tensors and arena are
 * addressed through the RVV view of the CLP window (board.h) and touched only by RVV
 * and the AMU, so they need no cache maintenance; the scalar-written mailbox is
 * written back after every update (ag_board_mbox_sync). The bf16 graph output
 * tensor (dense NHWC) has its own slot at the fixed BOARD_OUT_TENSOR_ADDR, zeroed
 * before the run (the mailbox repeats address and size), so the host can read it
 * back and judge the error on it as well as on the result frame. All text goes to
 * the UART; per-stage and per-node lines only in debug images (BOARD_LOG).
 * Release images mark progress instead: "#<stage>;" and "@<node>;" are written
 * only while the UART transmit FIFO is empty (never waiting, outside the node
 * timing), so after a hang the last marker shows where the program stopped.
 * After the done status the program prints the per-node cycles of the forward
 * and runs the extra stage (the main result stays valid whatever happens there):
 * arena cleared, a second forward with ag_conv_overlap = 0 and the graph output
 * at its arena offset, its per-node cycles, its result frame compared byte for
 * byte with the main one, the micro-benchmarks of tests/board_bench.c; it ends
 * with "extra done". */

#define ST_BOOT 0xB0000001u
#define ST_RUNNING 0xB0000003u   /* as the reference project; 0xB0000002 unused */
#define ST_DONE 0xB0000004u
#define ST_ERROR 0xB00000EEu

typedef struct {
    uint32_t status, error, cpu_hz, nodes;           /* 0x00 */
    uint64_t pre, forward, conv, other, post;        /* 0x10: cycles */
    uint64_t out_tensor, out_bytes;                  /* 0x38: bf16 graph output tensor (DDR view) */
    uint64_t mcause, mepc, mtval, mstatus, ra, sp;   /* 0x48: written by the trap handler */
    uint32_t stage, node;                            /* 0x78: include/animegan/board_log.h */
    uint32_t host_magic, echo;                       /* 0x80: written by the host, echoed by the board */
    uint8_t reserved[0x1000 - 0x88];
} ag_mailbox;
typedef char ag_mailbox_layout_check[(sizeof(ag_mailbox) == 0x1000 && offsetof(ag_mailbox, host_magic) == BOARD_MBOX_MAGIC - BOARD_MBOX_ADDR &&
                                      offsetof(ag_mailbox, mcause) == BOARD_MBOX_TRAP - BOARD_MBOX_ADDR &&
                                      offsetof(ag_mailbox, stage) == BOARD_MBOX_STAGE - BOARD_MBOX_ADDR) ? 1 : -1];
typedef char ag_out_tensor_slot_check[(BOARD_OUT_TENSOR_ADDR == BOARD_CLP_DDR + BOARD_CLP_OUT_TENSOR && BOARD_OUT_TENSOR_BYTES == 2 * BOARD_FRAME_BYTES &&
                                       BOARD_OUT_TENSOR_BYTES <= BOARD_CLP_ARENA - BOARD_CLP_OUT_TENSOR) ? 1 : -1];

extern const uint8_t ag_model_blob_start[], ag_model_blob_end[];
void ag_board_uart_init(void);
int ag_board_try_puts(const char *text);
uint64_t ag_board_cycles(void);
void ag_board_mbox_sync(void);
void ag_board_bench(uint8_t *work, size_t bytes);

#define CLP(off) ((void *)(uintptr_t)(BOARD_CLP_BASE + (off)))   /* RVV view */

#ifndef BOARD_BACKEND
#define BOARD_BACKEND AG_BACKEND_AME
#endif
#ifndef BOARD_BACKEND_STR
#define BOARD_BACKEND_STR "ame"
#endif
#if BOARD_DEBUG
#define BOARD_BUILD_STR "debug"
#else
#define BOARD_BUILD_STR "release"
#endif

static ag_mailbox *const mbox = (ag_mailbox *)(uintptr_t)BOARD_MBOX_ADDR;
static ag_model model;
static ag_runtime runtime;
static ag_plan plan;
static void **tensor_data;
static size_t *tensor_size;
static uint64_t *offsets;
static uint8_t *arena;
static uint16_t *input, *output;
static uint32_t graph_in = UINT32_MAX, graph_out = UINT32_MAX;
static uint64_t *node_cycles[2];      /* per node: [0] main forward, [1] extra forward */

#define uputs ag_plat_puts
#define uint_ ag_plat_put_int
static void uhex(uint64_t v) {
    int sh;
    char buf[19];
    buf[0] = '0'; buf[1] = 'x'; buf[18] = 0;
    for (sh = 0; sh < 16; ++sh) { int d = (int)(v >> (60 - 4 * sh)) & 15; buf[2 + sh] = (char)(d < 10 ? '0' + d : 'a' + d - 10); }
    uputs(buf);
}

/* progress marker "<tag><v>;" (release images), dropped while the UART is busy */
static void mark(char tag, uint32_t v) {
#if !BOARD_DEBUG
    char buf[16], digits[10];
    int n = 0, k = 0;
    do { digits[k++] = (char)('0' + v % 10); v /= 10; } while (v);
    buf[n++] = tag;
    while (k) buf[n++] = digits[--k];
    buf[n++] = ';';
    buf[n] = 0;
    ag_board_try_puts(buf);
#else
    (void)tag; (void)v;
#endif
}

static void set_stage(uint32_t stage) {
    ag_board_stage = stage;
    mark('#', stage);
    BOARD_LOG(uputs("stage "), uint_(stage), uputs(" cycle "), uint_((long)ag_board_cycles()), uputs("\n"));
}
/* The status word reaches DDR only after everything before it: the RVV result
 * frame / output tensor (CLP window) and the other mailbox fields, which span two
 * cache lines, are ordered and written back first (fence + l1d_clean_all +
 * fence), then the status word is written and written back the same way. */
static void mb_status(uint32_t s) {
    mbox->stage = ag_board_stage;
    mbox->node = ag_board_node;
    ag_board_mbox_sync();
    mbox->status = s;
    ag_board_mbox_sync();
}
static void fail(uint32_t code, const char *why, long detail) {
    mbox->error = code;
    uputs("error "); uint_(code); uputs(": "); uputs(why); uputs(" "); uint_(detail);
    uputs(" (stage "); uint_(ag_board_stage); uputs(" node "); uint_(ag_board_node); uputs(")\n");
    mb_status(ST_ERROR);
    for (;;) __asm__ volatile("wfi");
}

#ifdef AG_AME
/* Boot self-test of the RVV <-> AMU hand-over through the CLP window, on the
 * first 8 KB of the arena slot (zeroed afterwards): RVV writes A and B (32 x 32
 * bf16 ones) and clears C through the RVV view, fence, the AMU computes
 * C = A . B^T through the DDR view (ag_plat_ame_addr), fence, RVV reads C back.
 * Every element must be 32. C is cleared before the AMU writes it and A/B are
 * cleared again afterwards (pass or fail), so a late store in either direction
 * can never meet the ones or the 32s of an earlier boot. Returns 0, or 1 + the
 * index of the first bad element with its value in *bad. */
static long clp_selftest(float *bad) {
    uint16_t *ab = CLP(BOARD_CLP_ARENA);                  /* A = ab[0..1023], B = ab[1024..2047] */
    float *c = (float *)(ab + 2 * 32 * 32);
    size_t i = 0, vl;
    long first, result = 0;
    while (i < 2 * 32 * 32) { vl = __riscv_vsetvl_e16m8(2 * 32 * 32 - i); __riscv_vse16_v_u16m8(ab + i, __riscv_vmv_v_x_u16m8(0x3F80, vl), vl); i += vl; }
    memset(c, 0, 32 * 32 * sizeof(float));
    ag_plat_ame_sync_in(ab, 0);
    ag_ame_set_tile(32, 32, AG_AME_TILE_K);
    AG_AME_ZERO("acc0");
    ag_ame_macc1(ag_plat_ame_addr(ab + 32 * 32), AG_BLOCK * 2, ag_plat_ame_addr(ab), AG_BLOCK * 2);
    AG_AME_STORE_C("acc0", ag_plat_ame_addr(c), 32 * 4);
    ag_plat_ame_sync_out(c, 0);
    ag_plat_ame_release();
    for (i = 0; i < 32 * 32; i += vl) {
        vfloat32m8_t v;
        vl = __riscv_vsetvl_e32m8(32 * 32 - i);
        v = __riscv_vle32_v_f32m8(c + i, vl);
        first = __riscv_vfirst_m_b4(__riscv_vmfne_vf_f32m8_b4(v, 32.0f, vl), vl);
        if (first >= 0) {
            *bad = __riscv_vfmv_f_s_f32m8_f32(__riscv_vslidedown_vx_f32m8(v, (size_t)first, vl));
            result = 1 + (long)i + first;
            break;
        }
    }
    memset(ab, 0, 2 * 32 * 32 * sizeof(uint16_t));       /* RVV stores */
    ag_plat_ame_sync_in(ab, 0);
    return result;
}
#endif

static void pre_process(uint16_t *dst, const uint8_t *src, size_t n) {
#ifdef AG_RVV
    ag_rvv_u8_to_bf16(dst, src, n);
#else
    size_t i;
    for (i = 0; i < n; ++i) dst[i] = ag_f32_to_bf16((float)src[i] / 127.5f - 1.0f);
#endif
}

static void post_process(uint8_t *dst, const uint16_t *src, size_t n) {
#ifdef AG_RVV
    ag_rvv_bf16_to_u8(dst, src, n);
#else
    size_t i;
    for (i = 0; i < n; ++i) {
        float v = (ag_bf16_to_f32(src[i]) + 1.0f) * 127.5f;
        dst[i] = (uint8_t)(v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v);
    }
#endif
}

/* One forward pass over the static arena. Pass 0 (main): the graph output goes to
 * its fixed slot instead of its arena offset; pass 1 (extra): to its arena offset,
 * so the main result stays, and every node's output statistics are printed
 * ("stats node ...", tensor_io.h). Fills node_cycles[pass], cyc_conv / cyc_other;
 * only ag_runtime_execute_node is inside the timed interval. */
static int forward(int pass, uint64_t *cyc_conv, uint64_t *cyc_other) {
    uint32_t node_id, t;
    *cyc_conv = *cyc_other = 0;
    if (ag_runtime_init(&runtime, &model, tensor_data, tensor_size, model.tensor_count, BOARD_BACKEND) != 0) return -8;
    if (ag_runtime_bind(&runtime, graph_in, input, ag_tensor_bytes(model.tensors + graph_in)) != 0) return -9;
    runtime.scratch = arena + plan.scratch_offset;
    runtime.scratch_size = (size_t)plan.scratch_bytes;
    for (t = 0; t < model.tensor_count; ++t)
        if ((pass || t != graph_out) && offsets[t] != UINT64_MAX && ag_runtime_bind(&runtime, t, arena + offsets[t], ag_tensor_bytes(model.tensors + t)) != 0) return -9;
    if (!pass && ag_runtime_bind(&runtime, graph_out, output, BOARD_OUT_TENSOR_BYTES) != 0) return -9;
    for (node_id = 0; node_id < model.node_count; ++node_id) {
        const ag_node_desc *node = model.nodes + node_id;
        uint64_t c0;
        int rc;
        ag_board_node = node_id;
        ag_board_opcode = node->opcode;
        mark('@', node_id);
        c0 = ag_board_cycles();
        rc = ag_runtime_execute_node(&runtime, node_id);
        c0 = ag_board_cycles() - c0;
        node_cycles[pass][node_id] = c0;
        if (node->opcode == 3) *cyc_conv += c0; else *cyc_other += c0;
        BOARD_LOG(uputs("node "), uint_(node_id), uputs(" op "), uint_(node->opcode), uputs(" cycles "), uint_((long)c0), uputs("\n"));
        if (rc != 0) {
            uputs("node "); uint_(node_id); uputs(" opcode "); uint_(node->opcode); uputs(" failed rc="); uint_(rc); uputs("\n");
            return -10;
        }
        if (pass) {
            const int32_t *outputs = ag_model_at(&model, node->output_offset, sizeof(int32_t));
            ag_stats st;
            if (!outputs) return -11;
            ag_tensor_stats(model.tensors + outputs[0], tensor_data[outputs[0]], &st);
            ag_print_stats(node_id, &st);
        }
    }
    return 0;
}

static void print_node_cycles(int pass) {
    uint32_t i;
    uputs(pass ? "per-node cycles, extra forward (overlap off):\n" : "per-node cycles, main forward:\n");
    for (i = 0; i < model.node_count; ++i) {
        uputs("node "); uint_(i); uputs(" op "); uint_(model.nodes[i].opcode); uputs(" cycles "); uint_((long)node_cycles[pass][i]); uputs("\n");
    }
}

/* number of bytes in which a and b differ (RVV: frames are CLP data); *first = index of the first or -1 */
static size_t bytes_differ(const uint8_t *a, const uint8_t *b, size_t n, long *first) {
    size_t count = 0, i = 0;
    *first = -1;
#ifdef AG_RVV
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m8(n - i);
        vbool1_t ne = __riscv_vmsne_vv_u8m8_b1(__riscv_vle8_v_u8m8(a + i, vl), __riscv_vle8_v_u8m8(b + i, vl), vl);
        long f = __riscv_vfirst_m_b1(ne, vl);
        if (f >= 0 && *first < 0) *first = (long)i + f;
        count += __riscv_vcpop_m_b1(ne, vl);
        i += vl;
    }
#else
    for (; i < n; ++i) if (a[i] != b[i]) { if (*first < 0) *first = (long)i; ++count; }
#endif
    return count;
}

/* Extra stage, after the done status: nothing here changes the main result. */
static void extra(const uint8_t *frame_out) {
    uint64_t conv, other, c0, frame2_off = (plan.arena_bytes + 63) & ~(uint64_t)63;
    int rc;
    set_stage(AG_STAGE_EXTRA);
    if (offsets[graph_out] == UINT64_MAX) uputs("extra: graph output has no arena offset, second forward skipped\n");
    else {
        memset(arena, 0, (size_t)plan.arena_bytes);   /* RVV stores (libc_min) */
#ifdef AG_AME
        ag_conv_overlap = 0;
#endif
        c0 = ag_board_cycles();
        rc = forward(1, &conv, &other);
        c0 = ag_board_cycles() - c0;
#ifdef AG_AME
        ag_conv_overlap = 1;
#endif
        uputs("\nextra cycles conv="); uint_((long)conv); uputs(" other="); uint_((long)other);
        uputs(" (overlap off; node sums: the pass total "); uint_((long)c0); uputs(" includes the statistics output)\n");
        if (rc != 0) { uputs("extra: second forward failed rc="); uint_(rc); uputs("\n"); }
        else {
            print_node_cycles(1);
            if (frame2_off + BOARD_FRAME_BYTES > BOARD_CLP_BYTES - BOARD_CLP_ARENA) uputs("extra: no CLP room for the second frame, comparison skipped\n");
            else {
                long first;
                size_t n;
                post_process(arena + frame2_off, (const uint16_t *)(arena + offsets[graph_out]), BOARD_FRAME_BYTES);
                n = bytes_differ(arena + frame2_off, frame_out, BOARD_FRAME_BYTES, &first);
                uputs("extra: second forward result frame vs main result frame: ");
                if (!n) uputs("identical\n");
                else { uputs("differ in "); uint_((long)n); uputs(" bytes, first at byte "); uint_(first); uputs("\n"); }
            }
        }
    }
    set_stage(AG_STAGE_BENCH);
    ag_board_bench(arena, (size_t)plan.arena_bytes);
    /* trap self-test: the handler recognises the flag, prints "trap self-test ok" and
     * resumes after the 32-bit ebreak; a handler that does not prints a breakpoint
     * trap report and halts (no "extra done") */
    set_stage(AG_STAGE_TRAPTEST);
    ag_trap_selftest = 1;
    __asm__ volatile(".option push\n\t.option norvc\n\tebreak\n\t.option pop" ::: "memory");
    if (ag_trap_selftest != 2) uputs("trap self-test: resumed without the handler\n");
    ag_trap_selftest = 0;
    uputs("extra done\n");
}

void ag_board_main(void) {
    const uint8_t *frame_in = CLP(BOARD_CLP_INPUT);
    uint8_t *frame_out = CLP(BOARD_CLP_OUTPUT);
    uint64_t c0;
    uint32_t t, magic;
    size_t in_bytes;
    int rc;
    ag_board_stage = AG_STAGE_BOOT;
    ag_board_uart_init();
    magic = mbox->host_magic;             /* the cache is empty after reset: DDR contents */
    memset(mbox, 0, offsetof(ag_mailbox, host_magic));
    mbox->echo = magic;
    mbox->host_magic = 0;                 /* a frame without a fresh magic fails */
    mbox->cpu_hz = BOARD_CPU_HZ;
    mb_status(ST_BOOT);
    __asm__ volatile("csrw frm, x0");     /* round to nearest even for the fp32 arithmetic */
    uputs("\nanimeganv3 backdoor " BOARD_NAME " backend=" BOARD_BACKEND_STR " bf16 " BOARD_BUILD_STR "\n");
    uputs("pcie magic "); uhex(magic);
    if (magic != BOARD_HOST_MAGIC) { uputs(" mismatch\n"); fail(1, "pcie magic (host writes 0x5A5AC3C3 at 0x87F00080 before reset), got", (long)magic); }
    uputs(" ok\n");
#ifdef AG_AME
    set_stage(AG_STAGE_SELFTEST);
    {
        union { float f; uint32_t u; } bad = { 0.0f };
        long idx = clp_selftest(&bad.f);
        if (idx) {
            uputs("clp self-test: C["); uint_(idx - 1); uputs("] bits="); uhex(bad.u); uputs(", want 32.0 (bits=0x42000000)\n");
            fail(8, "clp self-test (RVV window -> AMU -> RVV window), first bad element", idx - 1);
        }
        uputs("clp self-test ok\n");
    }
#endif
    set_stage(AG_STAGE_MODEL);
    if (ag_model_open(&model, ag_model_blob_start, (size_t)(ag_model_blob_end - ag_model_blob_start)) != 0) fail(2, "model open", 0);
    tensor_data = ag_plat_alloc(model.tensor_count * sizeof(void *));
    tensor_size = ag_plat_alloc(model.tensor_count * sizeof(size_t));
    offsets = ag_plat_alloc(model.tensor_count * sizeof(uint64_t));
    node_cycles[0] = ag_plat_alloc(2 * model.node_count * sizeof(uint64_t));
    node_cycles[1] = node_cycles[0] + model.node_count;
    if (!tensor_data || !tensor_size || !offsets || !node_cycles[0]) fail(3, "alloc", 0);
    for (t = 0; t < model.tensor_count; ++t) {
        if (model.tensors[t].flags & 2) graph_in = t;
        if (model.tensors[t].flags & 4) graph_out = t;
    }
    if (graph_in == UINT32_MAX || graph_out == UINT32_MAX) fail(4, "graph io missing", 0);
    if (model.tensors[graph_in].type != AG_TYPE_BF16 || model.tensors[graph_out].type != AG_TYPE_BF16) fail(4, "graph io not bf16", 0);
    if (ag_tensor_elements(model.tensors + graph_in) != BOARD_FRAME_BYTES || ag_tensor_elements(model.tensors + graph_out) != BOARD_FRAME_BYTES)
        fail(4, "graph io is not a 512x512 RGB frame", (long)ag_tensor_elements(model.tensors + graph_in));
    if (ag_tensor_bytes(model.tensors + graph_out) != BOARD_OUT_TENSOR_BYTES)
        fail(4, "graph output is not the dense bf16 tensor of its fixed slot, bytes", (long)ag_tensor_bytes(model.tensors + graph_out));
    if ((rc = ag_plan_build(&model, BOARD_BACKEND, offsets, &plan)) != 0) fail(5, "plan", rc);
    in_bytes = ag_tensor_bytes(model.tensors + graph_in) + AG_PLAN_TAIL;
    if (in_bytes > BOARD_CLP_OUT_TENSOR - BOARD_CLP_GRAPH_IN) fail(3, "graph input does not fit its CLP slot", (long)in_bytes);
    if (plan.arena_bytes > BOARD_CLP_BYTES - BOARD_CLP_ARENA) fail(3, "arena does not fit the CLP window", (long)plan.arena_bytes);
    ag_board_arena_bytes = plan.arena_bytes;      /* for the trap report's CLP offsets (scratch at arena offset 0, planner.c) */
    ag_board_scratch_bytes = plan.scratch_bytes;
    input = CLP(BOARD_CLP_GRAPH_IN);
    output = CLP(BOARD_CLP_OUT_TENSOR);
    arena = CLP(BOARD_CLP_ARENA);
    mbox->nodes = model.node_count;
    BOARD_LOG(uputs("model nodes="), uint_(model.node_count), uputs(" tensors="), uint_(model.tensor_count),
              uputs(" arena="), uint_((long)plan.arena_bytes), uputs(" at "), uhex((uintptr_t)ag_plat_ame_addr(arena)),
              uputs(" scratch="), uint_((long)plan.scratch_bytes), uputs("\n"));
    set_stage(AG_STAGE_ZERO);
    memset(input, 0, in_bytes);           /* RVV stores (libc_min) */
    memset(output, 0, BOARD_OUT_TENSOR_BYTES);   /* a run that stops early leaves zeros, not the last frame's tensor */
    memset(arena, 0, (size_t)plan.arena_bytes);
    mb_status(ST_RUNNING);
    set_stage(AG_STAGE_PRE);
    c0 = ag_board_cycles();
    pre_process(input, frame_in, BOARD_FRAME_BYTES);
    mbox->pre = ag_board_cycles() - c0;
    set_stage(AG_STAGE_FORWARD);
    c0 = ag_board_cycles();
    rc = forward(0, &mbox->conv, &mbox->other);
    mbox->forward = ag_board_cycles() - c0;
    if (rc != 0) fail(6, "forward", rc);
    mbox->out_tensor = (uint64_t)(uintptr_t)ag_plat_ame_addr(tensor_data[graph_out]);   /* DDR view */
    mbox->out_bytes = tensor_size[graph_out];
    BOARD_LOG(uputs("output tensor "), uhex(mbox->out_tensor), uputs(" bytes "), uint_((long)mbox->out_bytes), uputs("\n"));
    set_stage(AG_STAGE_POST);
    c0 = ag_board_cycles();
    post_process(frame_out, tensor_data[graph_out], BOARD_FRAME_BYTES);
    mbox->post = ag_board_cycles() - c0;
    ag_board_stage = AG_STAGE_DONE;
    uputs("cycles pre="); uint_((long)mbox->pre); uputs(" forward="); uint_((long)mbox->forward);
    uputs(" conv="); uint_((long)mbox->conv); uputs(" other="); uint_((long)mbox->other);
    uputs(" post="); uint_((long)mbox->post); uputs(" cpu_hz="); uint_(BOARD_CPU_HZ); uputs("\ndone\n");
    mb_status(ST_DONE);
    print_node_cycles(0);
    extra(frame_out);
    for (;;) __asm__ volatile("wfi");
}
