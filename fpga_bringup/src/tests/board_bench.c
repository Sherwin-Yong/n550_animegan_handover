#include <animegan/ame.h>
#include <animegan/platform.h>
#include <animegan/runtime.h>
#include <stdint.h>
#include <string.h>
#ifdef AG_AME
#include <riscv_vector.h>
#endif

/* Micro-benchmarks of the board program's extra stage (tests/board_backdoor.c),
 * run after the main result is written. `work` is a CLP buffer (RVV view, the
 * AMU gets ag_plat_ame_addr) of at least BENCH_BYTES; it is zeroed first. All
 * counts are mcycle (BOARD_CPU_HZ); "x100" values are hundredths. One line per
 * result, "bench B<n> ...":
 *   B2 cycles per RVV instruction: 8 independent instructions per loop
 *      iteration, 256 iterations, for the instruction kinds of the kernels.
 *   B3 convolution epilogue cycles per output row: the form before S02
 *      (13 instructions, integer bf16 rounding, compare + merge) and the current
 *      ag_conv_epilogue_rvv, KO = 32 and 64, fused LeakyReLU, 384 rows.
 *   B4 AMU / RVV overlap, 16 groups of 3 tiles (128 x 32, 9 matrix steps): the
 *      matrix steps alone (a), the 384-row epilogue alone (b), and both
 *      interleaved as in ag_conv2d_ame (c); overlap works when c is close to
 *      max(a, b) rather than a + b.
 *   B5 the matrix part of B4 with mtilen = 128 and with mtilen = 32. */
void ag_board_bench(uint8_t *work, size_t bytes);
uint64_t ag_board_cycles(void);

#define uputs ag_plat_puts
#define uint_ ag_plat_put_int
#define BENCH_BYTES (1u << 20)

#ifdef AG_AME
#define ITER 256
#define VCLOB "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"

static void report_x100(const char *what, uint64_t cycles, uint64_t count) {
    uputs(what); uint_((long)(cycles * 100 / (count ? count : 1))); uputs("\n");
}

/* ---- B2: each kind in its own function, so no vector state crosses into compiled code ---- */
#define B2_KIND(fn, vset, body)                                                                     \
    __attribute__((noinline)) static uint64_t fn(void *p) {                                         \
        uint64_t c0;                                                                                \
        long it;                                                                                    \
        __asm__ volatile(vset ::: "t0", "memory");                                                  \
        c0 = ag_board_cycles();                                                                     \
        for (it = 0; it < ITER; ++it) __asm__ volatile(body :: "r"(p) : "t1", "memory", VCLOB);     \
        return ag_board_cycles() - c0;                                                              \
    }
#define M1_SET "li t0, 32\n\tvsetvli t0, t0, e32, m1, ta, ma"
#define MF2_SET "li t0, 32\n\tvsetvli t0, t0, e16, mf2, ta, ma"
#define M8_SET "li t0, 256\n\tvsetvli t0, t0, e32, m8, ta, ma"
#define M4H_SET "li t0, 256\n\tvsetvli t0, t0, e16, m4, ta, ma"
#define M8H_SET "li t0, 512\n\tvsetvli t0, t0, e16, m8, ta, ma"
B2_KIND(b2_vsetvli, M1_SET, "vsetvli t1, zero, e32, m1, ta, ma\n\tvsetvli t1, zero, e32, m1, ta, ma\n\tvsetvli t1, zero, e32, m1, ta, ma\n\t"
                            "vsetvli t1, zero, e32, m1, ta, ma\n\tvsetvli t1, zero, e32, m1, ta, ma\n\tvsetvli t1, zero, e32, m1, ta, ma\n\t"
                            "vsetvli t1, zero, e32, m1, ta, ma\n\tvsetvli t1, zero, e32, m1, ta, ma")
B2_KIND(b2_vfadd_m1, M1_SET, "vfadd.vv v1, v9, v17\n\tvfadd.vv v2, v10, v18\n\tvfadd.vv v3, v11, v19\n\tvfadd.vv v4, v12, v20\n\t"
                             "vfadd.vv v5, v13, v21\n\tvfadd.vv v6, v14, v22\n\tvfadd.vv v7, v15, v23\n\tvfadd.vv v8, v16, v24")
B2_KIND(b2_vfmul_m1, M1_SET, "vfmul.vf v1, v9, ft0\n\tvfmul.vf v2, v10, ft0\n\tvfmul.vf v3, v11, ft0\n\tvfmul.vf v4, v12, ft0\n\t"
                             "vfmul.vf v5, v13, ft0\n\tvfmul.vf v6, v14, ft0\n\tvfmul.vf v7, v15, ft0\n\tvfmul.vf v8, v16, ft0")
B2_KIND(b2_vfmax_m1, M1_SET, "vfmax.vv v1, v9, v17\n\tvfmax.vv v2, v10, v18\n\tvfmax.vv v3, v11, v19\n\tvfmax.vv v4, v12, v20\n\t"
                             "vfmax.vv v5, v13, v21\n\tvfmax.vv v6, v14, v22\n\tvfmax.vv v7, v15, v23\n\tvfmax.vv v8, v16, v24")
B2_KIND(b2_vle32_m1, M1_SET, "vle32.v v1, (%0)\n\tvle32.v v2, (%0)\n\tvle32.v v3, (%0)\n\tvle32.v v4, (%0)\n\t"
                             "vle32.v v5, (%0)\n\tvle32.v v6, (%0)\n\tvle32.v v7, (%0)\n\tvle32.v v8, (%0)")
B2_KIND(b2_vse16_mf2, MF2_SET, "vse16.v v1, (%0)\n\tvse16.v v2, (%0)\n\tvse16.v v3, (%0)\n\tvse16.v v4, (%0)\n\t"
                               "vse16.v v5, (%0)\n\tvse16.v v6, (%0)\n\tvse16.v v7, (%0)\n\tvse16.v v8, (%0)")
B2_KIND(b2_vfncvt_mf2, MF2_SET, "vfncvtbf16.f.f.w v1, v9\n\tvfncvtbf16.f.f.w v2, v10\n\tvfncvtbf16.f.f.w v3, v11\n\tvfncvtbf16.f.f.w v4, v12\n\t"
                                "vfncvtbf16.f.f.w v5, v13\n\tvfncvtbf16.f.f.w v6, v14\n\tvfncvtbf16.f.f.w v7, v15\n\tvfncvtbf16.f.f.w v8, v16")
B2_KIND(b2_vfwcvt_mf2, MF2_SET, "vfwcvtbf16.f.f.v v1, v17\n\tvfwcvtbf16.f.f.v v2, v18\n\tvfwcvtbf16.f.f.v v3, v19\n\tvfwcvtbf16.f.f.v v4, v20\n\t"
                                "vfwcvtbf16.f.f.v v5, v21\n\tvfwcvtbf16.f.f.v v6, v22\n\tvfwcvtbf16.f.f.v v7, v23\n\tvfwcvtbf16.f.f.v v8, v24")
B2_KIND(b2_vfadd_m8, M8_SET, "vfadd.vv v8, v8, v16\n\tvfadd.vv v24, v24, v16\n\tvfadd.vv v8, v8, v16\n\tvfadd.vv v24, v24, v16\n\t"
                             "vfadd.vv v8, v8, v16\n\tvfadd.vv v24, v24, v16\n\tvfadd.vv v8, v8, v16\n\tvfadd.vv v24, v24, v16")
B2_KIND(b2_vle32_m8, M8_SET, "vle32.v v8, (%0)\n\tvle32.v v16, (%0)\n\tvle32.v v24, (%0)\n\tvle32.v v8, (%0)\n\t"
                             "vle32.v v16, (%0)\n\tvle32.v v24, (%0)\n\tvle32.v v8, (%0)\n\tvle32.v v16, (%0)")
B2_KIND(b2_vse16_m4, M4H_SET, "vse16.v v8, (%0)\n\tvse16.v v12, (%0)\n\tvse16.v v16, (%0)\n\tvse16.v v20, (%0)\n\t"
                              "vse16.v v8, (%0)\n\tvse16.v v12, (%0)\n\tvse16.v v16, (%0)\n\tvse16.v v20, (%0)")
B2_KIND(b2_vfncvt_m4, M4H_SET, "vfncvtbf16.f.f.w v8, v16\n\tvfncvtbf16.f.f.w v12, v24\n\tvfncvtbf16.f.f.w v8, v16\n\tvfncvtbf16.f.f.w v12, v24\n\t"
                               "vfncvtbf16.f.f.w v8, v16\n\tvfncvtbf16.f.f.w v12, v24\n\tvfncvtbf16.f.f.w v8, v16\n\tvfncvtbf16.f.f.w v12, v24")
B2_KIND(b2_vrgather_m8, M8H_SET, "vrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24\n\t"
                                 "vrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24\n\tvrgather.vv v8, v16, v24")

static void bench_b2(void *p) {
    static const struct { const char *name; uint64_t (*fn)(void *); } kinds[] = {
        {"vsetvli", b2_vsetvli}, {"vfadd.vv e32m1", b2_vfadd_m1}, {"vfmul.vf e32m1", b2_vfmul_m1}, {"vfmax.vv e32m1", b2_vfmax_m1},
        {"vle32.v e32m1 (128 B, CLP)", b2_vle32_m1}, {"vse16.v e16mf2 (64 B, CLP)", b2_vse16_mf2},
        {"vfncvtbf16 e16mf2", b2_vfncvt_mf2}, {"vfwcvtbf16 e16mf2", b2_vfwcvt_mf2}, {"vfadd.vv e32m8", b2_vfadd_m8},
        {"vle32.v e32m8 (1 KB, CLP)", b2_vle32_m8}, {"vse16.v e16m4 (512 B, CLP)", b2_vse16_m4}, {"vfncvtbf16 e16m4", b2_vfncvt_m4},
        {"vrgather.vv e16m8", b2_vrgather_m8},
    };
    unsigned i;
    for (i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        uint64_t c = kinds[i].fn(p);
        uputs("bench B2 "); uputs(kinds[i].name); report_x100(" cycles/insn x100 ", c, 8 * ITER);
    }
}

/* ---- B3: the epilogue before S02, per row: vle32, vfadd, vmflt, vfmul, vmerge,
 * vfmerge, then the integer round to bf16 (vsrl, vand, vadd, vadd, vnsrl) and vse16 ---- */
__attribute__((noinline)) static void epi_old(uint16_t *y_row0, size_t plane_elems, const float *cbuf, int32_t rows, int32_t KO,
                                              int32_t blocks, const float *bias, float alpha) {
    int32_t cb, r;
    for (cb = 0; cb < blocks; ++cb) {
        int32_t lanes = KO - cb * AG_BLOCK < AG_BLOCK ? KO - cb * AG_BLOCK : AG_BLOCK;
        size_t vl = __riscv_vsetvl_e32m1(AG_BLOCK);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(bias + cb * AG_BLOCK, (size_t)lanes);
        vbool32_t valid = __riscv_vmsltu_vx_u32m1_b32(__riscv_vid_v_u32m1(vl), (uint32_t)lanes, vl);
        for (r = 0; r < rows; ++r) {
            vfloat32m1_t v = __riscv_vle32_v_f32m1(cbuf + (size_t)r * KO + cb * AG_BLOCK, (size_t)lanes);
            vuint32m1_t u, lsb;
            v = __riscv_vfadd_vv_f32m1(v, vb, vl);
            v = __riscv_vmerge_vvm_f32m1(v, __riscv_vfmul_vf_f32m1(v, alpha, vl), __riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl), vl);
            v = __riscv_vfmerge_vfm_f32m1(v, 0.0f, __riscv_vmnot_m_b32(valid, vl), vl);
            u = __riscv_vreinterpret_v_f32m1_u32m1(v);
            lsb = __riscv_vand_vx_u32m1(__riscv_vsrl_vx_u32m1(u, 16, vl), 1, vl);
            u = __riscv_vadd_vv_u32m1(__riscv_vadd_vx_u32m1(u, 0x7FFFu, vl), lsb, vl);
            __riscv_vse16_v_u16mf2(y_row0 + (size_t)cb * plane_elems + (size_t)r * AG_BLOCK, __riscv_vnsrl_wx_u16mf2(u, 16, vl), vl);
        }
    }
}

#define ROWS 384   /* one group of 3 tiles of 128 pixels */
static void epi_init(ag_conv_epi *e, int32_t KO, const float *bias) {
    ag_conv_geom g;
    memset(&g, 0, sizeof(g));
    g.KO = KO; g.out.C = KO; g.out.blocks = (KO + AG_BLOCK - 1) / AG_BLOCK; g.out.plane = ROWS;
    g.bias = bias; g.fused = AG_FUSED_LEAKY; g.alpha = 0.2f;
    ag_conv_epilogue_rvv_init(e, &g);
}

static void bench_b3(uint8_t *work) {
    static float bias[2 * AG_BLOCK];
    float *c = (float *)work;                               /* ROWS x 64 fp32 = 96 KB */
    uint16_t *y = (uint16_t *)(work + 128 * 1024);          /* 2 blocks x ROWS x 64 B = 48 KB */
    int32_t KO;
    for (KO = 32; KO <= 64; KO += 32) {
        ag_conv_epi e;
        uint64_t c_old, c_new;
        epi_init(&e, KO, bias);
        c_old = ag_board_cycles();
        epi_old(y, (size_t)ROWS * AG_BLOCK, c, ROWS, KO, KO / AG_BLOCK, bias, 0.2f);
        c_old = ag_board_cycles() - c_old;
        c_new = ag_board_cycles();
        ag_conv_epilogue_rvv(&e, y, c, ROWS);
        c_new = ag_board_cycles() - c_new;
        uputs("bench B3 epilogue KO="); uint_(KO);
        uputs(" cycles/row x100 old "); uint_((long)(c_old * 100 / ROWS)); uputs(" new "); uint_((long)(c_new * 100 / ROWS)); uputs("\n");
    }
}

/* ---- B4 / B5: matrix steps of one group of 3 tiles (TM = 128, N = n, 9 steps) ---- */
#define GROUPS 16
#define STEPS 9
static void group_matrix(const uint16_t *b, const uint16_t *a, float *cm, long n, ag_conv_epi *e, uint16_t *y, const float *c, int32_t slice) {
    int32_t s;
    AG_AME_ZERO("acc0");
    AG_AME_ZERO("acc1");
    AG_AME_ZERO("acc2");
    for (s = 0; s < STEPS; ++s) {
        ag_ame_macc3(b, AG_BLOCK * 2, a, a + 128 * AG_BLOCK, a + 2 * 128 * AG_BLOCK, AG_BLOCK * 2);
        if (slice) ag_conv_epilogue_rvv(e, y + (size_t)s * slice * AG_BLOCK, c + (size_t)s * slice * 32, s == STEPS - 1 ? ROWS - s * slice : slice);
    }
    __asm__ volatile("fence rw,rw" ::: "memory");
    AG_AME_STORE_C("acc0", cm, n * 4);
    AG_AME_STORE_C("acc1", cm + 128 * n, n * 4);
    AG_AME_STORE_C("acc2", cm + 2 * 128 * n, n * 4);
    __asm__ volatile("fence rw,rw" ::: "memory");
}

static uint64_t run_groups(uint8_t *work, long n, int matrix, int epilogue) {
    static float bias[AG_BLOCK];
    const uint16_t *a = ag_plat_ame_addr(work), *b = ag_plat_ame_addr(work + 32 * 1024);   /* A 3 x 8 KB, B up to 8 KB */
    float *cm = ag_plat_ame_addr(work + 64 * 1024);                                          /* C up to 192 KB */
    const float *c = (const float *)(work + 320 * 1024);                                     /* epilogue C: ROWS x 32 fp32 */
    uint16_t *y = (uint16_t *)(work + 384 * 1024);                                           /* ROWS x 64 B */
    ag_conv_epi e;
    uint64_t c0;
    int g;
    epi_init(&e, 32, bias);
    ag_ame_set_tile(128, n, AG_AME_TILE_K);
    c0 = ag_board_cycles();
    for (g = 0; g < GROUPS; ++g) {
        if (matrix) group_matrix(b, a, cm, n, &e, y, c, epilogue ? (ROWS + STEPS - 1) / STEPS : 0);
        else ag_conv_epilogue_rvv(&e, y, c, ROWS);
    }
    c0 = ag_board_cycles() - c0;
    ag_plat_ame_release();
    return c0;
}

static void bench_b4_b5(uint8_t *work) {
    uint64_t a = run_groups(work, 32, 1, 0), b = run_groups(work, 32, 0, 1), c = run_groups(work, 32, 1, 1), n128;
    uint64_t mx = a > b ? a : b;
    uputs("bench B4 groups="); uint_(GROUPS); uputs(" steps="); uint_(STEPS); uputs(" rows="); uint_(ROWS);
    uputs(" matrix "); uint_((long)a); uputs(" epilogue "); uint_((long)b); uputs(" interleaved "); uint_((long)c);
    uputs(" c/(a+b) x100 "); uint_((long)(c * 100 / (a + b ? a + b : 1))); uputs(" c/max(a,b) x100 "); uint_((long)(c * 100 / (mx ? mx : 1))); uputs("\n");
    n128 = run_groups(work, 128, 1, 0);
    uputs("bench B5 matrix groups="); uint_(GROUPS); uputs(" steps="); uint_(STEPS);
    uputs(" mtilen=128 "); uint_((long)n128); uputs(" mtilen=32 "); uint_((long)a); uputs("\n");
}

void ag_board_bench(uint8_t *work, size_t bytes) {
    if (bytes < BENCH_BYTES) { uputs("bench skipped: work area too small\n"); return; }
    memset(work, 0, BENCH_BYTES);          /* RVV stores (libc_min) */
    __asm__ volatile("fence rw,rw" ::: "memory");
    bench_b2(work);
    bench_b3(work);
    bench_b4_b5(work);
}
#else
void ag_board_bench(uint8_t *work, size_t bytes) { (void)work; (void)bytes; uputs("bench skipped: not an AME/RVV build\n"); }
#endif
