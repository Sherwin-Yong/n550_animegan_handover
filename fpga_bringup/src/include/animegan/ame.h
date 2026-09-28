#ifndef ANIMEGAN_AME_H
#define ANIMEGAN_AME_H

/* AME (matrix extension) instruction wrappers shared by the GEMM and the
 * convolution kernel. Tile geometry for bf16: mtilem <= 128 rows, mtilen <= 128
 * columns, mtilek = 32 bf16 (64 bytes per tile row).
 *
 * Hardware constraint (D35): the address and stride registers of a matrix
 * instruction must be among a0-a3, so every wrapper moves them into a0-a3
 * inside the same asm block as the instruction and clobbers those registers.
 * Tile CSRs (0x803 mtilem, 0x804 mtilen, 0x805 mtilek) are written with csrw
 * through t0. mfmacc.s.bf16 takes (acc, B tile, A tile): C += A . B^T. */
#define AG_AME_TILE_M 128
#define AG_AME_TILE_N 128
#define AG_AME_TILE_K 32

#define AG_AME_LOAD(insn, tr, ptr, stride)                                  \
    __asm__ volatile("mv a0, %0\n\t"                                        \
                     "mv a1, %1\n\t"                                        \
                     insn " " tr ",(a0),a1"                                 \
                     :: "r"(ptr), "r"((long)(stride)) : "a0", "a1", "memory")
#define AG_AME_STORE_C(acc, ptr, stride)                                    \
    __asm__ volatile("mv a0, %0\n\t"                                        \
                     "mv a1, %1\n\t"                                        \
                     "msce32 " acc ",(a0),a1"                               \
                     :: "r"(ptr), "r"((long)(stride)) : "a0", "a1", "memory")
#define AG_AME_ZERO(acc) __asm__ volatile("mzero " acc)
#define AG_AME_MACC(acc, trb, tra) __asm__ volatile("mfmacc.s.bf16 " acc "," trb "," tra)

/* One k-step of one to three M tiles sharing a B tile: B -> tr0 (address a0,
 * stride a1), A_i -> tr(1+i) (address a2, stride a3), acc_i += A_i . B^T. */
static inline void ag_ame_macc3(const void *b, long b_stride, const void *p0, const void *p1, const void *p2, long a_stride) {
    __asm__ volatile("mv a0, %0\n\t"
                     "mv a1, %1\n\t"
                     "mlbe16 tr0,(a0),a1\n\t"
                     "mv a3, %5\n\t"
                     "mv a2, %2\n\t"
                     "mlae16 tr1,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc0,tr0,tr1\n\t"
                     "mv a2, %3\n\t"
                     "mlae16 tr2,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc1,tr0,tr2\n\t"
                     "mv a2, %4\n\t"
                     "mlae16 tr3,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc2,tr0,tr3"
                     :: "r"(b), "r"(b_stride), "r"(p0), "r"(p1), "r"(p2), "r"(a_stride) : "a0", "a1", "a2", "a3", "memory");
}
static inline void ag_ame_macc2(const void *b, long b_stride, const void *p0, const void *p1, long a_stride) {
    __asm__ volatile("mv a0, %0\n\t"
                     "mv a1, %1\n\t"
                     "mlbe16 tr0,(a0),a1\n\t"
                     "mv a3, %4\n\t"
                     "mv a2, %2\n\t"
                     "mlae16 tr1,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc0,tr0,tr1\n\t"
                     "mv a2, %3\n\t"
                     "mlae16 tr2,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc1,tr0,tr2"
                     :: "r"(b), "r"(b_stride), "r"(p0), "r"(p1), "r"(a_stride) : "a0", "a1", "a2", "a3", "memory");
}
static inline void ag_ame_macc1(const void *b, long b_stride, const void *p0, long a_stride) {
    __asm__ volatile("mv a0, %0\n\t"
                     "mv a1, %1\n\t"
                     "mlbe16 tr0,(a0),a1\n\t"
                     "mv a3, %3\n\t"
                     "mv a2, %2\n\t"
                     "mlae16 tr1,(a2),a3\n\t"
                     "mfmacc.s.bf16 acc0,tr0,tr1"
                     :: "r"(b), "r"(b_stride), "r"(p0), "r"(a_stride) : "a0", "a1", "a2", "a3", "memory");
}

static inline void ag_ame_set_tile(long m, long n, long k) {
    __asm__ volatile("mv   t0, %0\n\t"
                     "csrw 0x803, t0\n\t"
                     "mv   t0, %1\n\t"
                     "csrw 0x804, t0\n\t"
                     "mv   t0, %2\n\t"
                     "csrw 0x805, t0"
                     :: "r"(m), "r"(n), "r"(k) : "t0");
}

/* round to nearest even for the fp32 accumulation (the reference vectors are RNE) */
static inline void ag_ame_rne(void) { __asm__ volatile("csrw frm, x0"); }

#endif
