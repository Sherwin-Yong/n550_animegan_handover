#include <animegan/platform.h>
#include <stdint.h>
#include "bsp/board.h"

/* Bare-metal platform layer: 16550 UART console, bump allocator over
 * BOARD_HEAP (scalar-only data), mcycle clock, fences at the AME boundary,
 * the RVV-window -> DDR address map for the AMU, mailbox write-back,
 * mstatus.MS release. There is no file system: the
 * model pack is linked into the image and the frames are written into DDR by
 * the host, so ag_plat_read_file/write_file report failure. */

#define UART_REG(idx) ((uintptr_t)BOARD_UART_BASE + ((uintptr_t)(idx) << BOARD_UART_REG_SHIFT))
#if BOARD_UART_32BIT
#define UART_RD(idx) (*(volatile uint32_t *)UART_REG(idx))
#define UART_WR(idx, v) (*(volatile uint32_t *)UART_REG(idx) = (uint32_t)(v))
#else
#define UART_RD(idx) (*(volatile uint8_t *)UART_REG(idx))
#define UART_WR(idx, v) (*(volatile uint8_t *)UART_REG(idx) = (uint8_t)(v))
#endif

static uint64_t rd_mcycle(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

void ag_board_uart_init(void) {
#if BOARD_UART_NEEDS_INIT
    /* divisor = clk / (16 * baud); the fraction goes to DW_apb_uart's DLF (1/2^bits) */
    uint32_t step = 16u * BOARD_UART_BAUD, div = BOARD_UART_CLK_HZ / step;
    uint32_t frac = ((BOARD_UART_CLK_HZ % step) * (1u << BOARD_UART_DLF_BITS) + step / 2) / step;
    if (frac >= (1u << BOARD_UART_DLF_BITS)) { frac = 0; ++div; }
    UART_WR(UART_IER, 0x00);
    UART_WR(UART_LCR, 0x80);
    UART_WR(UART_DLL, div & 0xFF);
    UART_WR(UART_DLM, (div >> 8) & 0xFF);
    UART_WR(UART_LCR, 0x03); /* 8N1, DLAB=0 */
#if BOARD_UART_HAS_DLF
    UART_WR(BOARD_UART_DLF_IDX, frac);
#endif
    UART_WR(UART_FCR, 0x07); /* FIFO on, clear */
#endif
}

void ag_board_putc(void *ctx, int c) {
    long spin = 0;
    (void)ctx;
    while (!(UART_RD(UART_LSR) & UART_LSR_THRE) && ++spin < 1000000) {}
    UART_WR(UART_THR, c & 0xFF);
}

/* ---- console ---- */
void ag_plat_puts(const char *text) {
    for (; *text; ++text) {
        if (*text == '\n') ag_board_putc(0, '\r');
        ag_board_putc(0, *text);
    }
}

void ag_plat_put_int(long value) {
    char buf[24];
    int pos = 23;
    unsigned long v = value < 0 ? (unsigned long)(-value) : (unsigned long)value;
    buf[pos] = 0;
    do { buf[--pos] = (char)('0' + v % 10); v /= 10; } while (v);
    if (value < 0) buf[--pos] = '-';
    ag_plat_puts(buf + pos);
}

long ag_plat_atol(const char *text) {
    long value = 0, sign = 1;
    if (*text == '-') { sign = -1; ++text; }
    while (*text >= '0' && *text <= '9') value = value * 10 + (*text++ - '0');
    return sign * value;
}

void ag_plat_exit(int code) {
#if BOARD_HAS_POWEROFF
    *(volatile uint32_t *)(uintptr_t)BOARD_POWEROFF_ADDR = code == 0 ? 0x5555u : 0x3333u;
#else
    (void)code;
#endif
    for (;;) __asm__ volatile("wfi");
}

/* ---- memory: bump allocator, zero-filled, 64 B aligned; free is a no-op ---- */
static uintptr_t g_heap_next = BOARD_HEAP_BASE;

void *ag_plat_alloc(size_t bytes) {
    uintptr_t base = (g_heap_next + BOARD_CACHE_LINE - 1) & ~(uintptr_t)(BOARD_CACHE_LINE - 1);
    uint64_t *p, *e;
    if (base + bytes > (uintptr_t)BOARD_HEAP_BASE + BOARD_HEAP_BYTES) return 0;
    g_heap_next = base + bytes;
    for (p = (uint64_t *)base, e = (uint64_t *)((base + bytes + 7) & ~(uintptr_t)7); p < e; ++p) *p = 0;
    return (void *)base;
}

void ag_plat_free(void *ptr) { (void)ptr; }
size_t ag_board_heap_used(void) { return (size_t)(g_heap_next - BOARD_HEAP_BASE); }

void *ag_plat_read_file(const char *path, size_t *size) { (void)path; (void)size; return 0; }
int ag_plat_write_file(const char *path, const void *data, size_t size) { (void)path; (void)data; (void)size; return -1; }

/* ---- time: mcycle; ns are exact on the board, arbitrary units on QEMU ---- */
long ag_plat_clock_ns(void) {
    uint64_t c = rd_mcycle();
    return BOARD_CPU_HZ ? (long)(c * (1000000000ull / BOARD_CPU_HZ)) : (long)c;
}
uint64_t ag_board_cycles(void) { return rd_mcycle(); }

/* ---- memory visibility ----
 * Tensor data is touched only by RVV through the uncached CLP window and by the
 * AMU (uncached, DDR view), so the RVV <-> AMU hand-over needs only ordering:
 * fence rw,rw. The weights are read-only and were written by the host while
 * the CPU was held in reset (empty cache). The only cached data the host reads
 * is the mailbox: after the CPU writes it, l1d_clean_all writes L1D back. */
void ag_board_mbox_sync(void) {
    __asm__ volatile("fence rw,rw" ::: "memory");
#if BOARD_L1D_ALL
    __asm__ volatile("l1d_clean_all" ::: "memory");
    __asm__ volatile("fence rw,rw" ::: "memory");
#endif
}

void ag_plat_ame_sync_in(const void *ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    __asm__ volatile("fence rw,rw" ::: "memory");
}

void ag_plat_ame_sync_out(const void *ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    __asm__ volatile("fence rw,rw" ::: "memory");
}

void *ag_plat_ame_addr(const void *ptr) {
    uintptr_t p = (uintptr_t)ptr;
    if (p - BOARD_CLP_BASE < BOARD_CLP_BYTES) p = p - BOARD_CLP_BASE + BOARD_CLP_DDR;
    return (void *)p;
}

/* mstatus.MS <- Clean (10) instead of mrelease (reference project: mrelease
 * corrupts the I-cache on this RTL). Read-modify-write keeps FS/VS. */
void ag_plat_ame_release(void) {
    unsigned long ms;
    __asm__ volatile("csrr %0, mstatus" : "=r"(ms));
    ms = (ms & ~(3UL << 29)) | (2UL << 29);
    __asm__ volatile("csrw mstatus, %0" ::"r"(ms) : "memory");
}
