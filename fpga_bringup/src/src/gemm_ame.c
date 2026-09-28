#include <animegan/ame.h>
#include <animegan/platform.h>
#include <animegan/runtime.h>

/* bf16 GEMM on the AME matrix unit: C[M][N] fp32 = A[M][K] . B[N][K]^T with
 * fp32 accumulation (mfmacc.s.bf16; the RTL rounds after every 16 products,
 * QEMU after 32). Tiles are mtilem = min(128, rows left), mtilen = N,
 * mtilek = 32 bf16; K must be a multiple of 32, 1 <= N <= 128. Strides are in
 * bytes. Loads read the full tile range, so every row of A and B must be
 * readable for K elements and the memory after the last row must stay
 * readable (the tile load over-reads about one row stride). Used by the GEMM
 * unit test; the convolution issues the same instruction sequence directly
 * (src/conv_ame.c). */
int ag_gemm_bf16_ame(float *c, long c_stride, const uint16_t *a, long a_stride,
                     const uint16_t *b, long b_stride, int M, int K, int N) {
    int m0, k0;
    if (!c || !a || !b) return -40;
    if (M <= 0) return -41;
    if (N <= 0 || N > AG_AME_TILE_N) return -42;
    if (K <= 0 || K % AG_AME_TILE_K) return -43;
    if (a_stride < K * 2 || b_stride < K * 2 || c_stride < (long)N * 4) return -44;
    ag_plat_ame_sync_in(a, (size_t)(M - 1) * a_stride + (size_t)K * 2);
    ag_plat_ame_sync_in(b, (size_t)(N - 1) * b_stride + (size_t)K * 2);
    ag_ame_rne();
    for (m0 = 0; m0 < M; m0 += AG_AME_TILE_M) {
        int rows = M - m0 < AG_AME_TILE_M ? M - m0 : AG_AME_TILE_M;
        const uint16_t *arow = a + (size_t)m0 * a_stride / 2;
        float *crow = (float *)((uint8_t *)c + (size_t)m0 * c_stride);
        ag_ame_set_tile(rows, N, AG_AME_TILE_K);
        AG_AME_ZERO("acc0");
        for (k0 = 0; k0 < K; k0 += AG_AME_TILE_K) {
            AG_AME_LOAD("mlae16", "tr0", arow + k0, a_stride);
            AG_AME_LOAD("mlbe16", "tr1", b + k0, b_stride);
            AG_AME_MACC("acc0", "tr1", "tr0");
        }
        AG_AME_STORE_C("acc0", crow, c_stride);
        ag_plat_ame_sync_out(crow, (size_t)(rows - 1) * c_stride + (size_t)N * 4);
    }
    ag_plat_ame_release();
    return 0;
}
