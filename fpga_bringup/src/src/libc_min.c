#include <stddef.h>
#include <stdint.h>

/* Freestanding libc pieces shared by the RV64 Linux user-mode platform layer
 * and the bare-metal board layer (both build with -nostdlib). RVV builds
 * (-DAG_RVV) copy/fill with vle8/vse8 strips, the scalar RV64
 * baseline keeps byte loops. */
/* Loop-distribution is off so GCC cannot turn these loops back into calls to
 * themselves. */
#define AG_NO_LOOP_PATTERNS __attribute__((optimize("no-tree-loop-distribute-patterns")))

#ifdef AG_RVV
/* RVV byte copy/fill: strip-mined vle8/vse8; the tail is handled
 * by vsetvl, so any length is exact. Only the AME/RVV build defines AG_RVV;
 * the scalar RV64 baseline keeps the byte loops below. */
#include <riscv_vector.h>
void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    while (n) {
        size_t vl = __riscv_vsetvl_e8m8(n);
        __riscv_vse8_v_u8m8(d, __riscv_vle8_v_u8m8(s, vl), vl);
        d += vl; s += vl; n -= vl;
    }
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    uint8_t *d = dst;
    while (n) {
        size_t vl = __riscv_vsetvl_e8m8(n);
        __riscv_vse8_v_u8m8(d, __riscv_vmv_v_x_u8m8((uint8_t)c, vl), vl);
        d += vl; n -= vl;
    }
    return dst;
}
#else
AG_NO_LOOP_PATTERNS void *memcpy(void *dst, const void *src, size_t n) {
    char *d = dst;
    const char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

AG_NO_LOOP_PATTERNS void *memset(void *dst, int c, size_t n) {
    char *d = dst;
    while (n--) *d++ = (char)c;
    return dst;
}
#endif

AG_NO_LOOP_PATTERNS int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    while (n--) {
        if (*x != *y) return *x - *y;
        ++x; ++y;
    }
    return 0;
}

size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) ++n;
    return n;
}

int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { ++a; ++b; }
    return (unsigned char)*a - (unsigned char)*b;
}

static int g_errno;
int *__errno(void) { return &g_errno; }

