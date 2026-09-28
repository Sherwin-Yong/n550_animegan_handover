#include <animegan/platform.h>
#include <stdint.h>

/* Freestanding platform layer for RV64 Linux user mode (qemu-riscv64).
 * The -elf toolchain's newlib routes IO through semihosting, which user-mode
 * QEMU does not translate, so files and console go straight to Linux
 * syscalls. Memory comes from anonymous mmap, one mapping per allocation.
 * memcpy/memset/strlen/... live in libc_min.c (shared with the board layer). */

#define SYS_openat 56
#define SYS_close 57
#define SYS_lseek 62
#define SYS_read 63
#define SYS_clock_gettime 113
#define SYS_write 64
#define SYS_exit 93
#define SYS_munmap 215
#define SYS_mmap 222

#define AT_FDCWD (-100)
#define O_RDONLY 0
#define O_WRONLY 1
#define O_CREAT 0100
#define O_TRUNC 01000
#define SEEK_SET 0
#define SEEK_END 2
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_ANONYMOUS 0x20

static long sc(long n, long a, long b, long c, long d, long e, long f) {
    register long a0 asm("a0") = a, a1 asm("a1") = b, a2 asm("a2") = c;
    register long a3 asm("a3") = d, a4 asm("a4") = e, a5 asm("a5") = f;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a7) : "memory");
    return a0;
}

static size_t page_round(size_t bytes) { return (bytes + 4095u) & ~(size_t)4095u; }

void *ag_plat_alloc(size_t bytes) {
    size_t total = page_round(bytes + 16);
    long r = sc(SYS_mmap, 0, (long)total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r < 0 && r > -4096) return 0;
    *(size_t *)r = total; /* anonymous pages are already zero */
    return (char *)r + 16;
}

void ag_plat_free(void *ptr) {
    char *base;
    if (!ptr) return;
    base = (char *)ptr - 16;
    sc(SYS_munmap, (long)base, (long)*(size_t *)base, 0, 0, 0, 0);
}

void *ag_plat_read_file(const char *path, size_t *size) {
    long fd = sc(SYS_openat, AT_FDCWD, (long)path, O_RDONLY, 0, 0, 0);
    long length, got = 0;
    char *data;
    if (fd < 0) return 0;
    length = sc(SYS_lseek, fd, 0, SEEK_END, 0, 0, 0);
    if (length < 0 || sc(SYS_lseek, fd, 0, SEEK_SET, 0, 0, 0) != 0) { sc(SYS_close, fd, 0, 0, 0, 0, 0); return 0; }
    data = ag_plat_alloc((size_t)length);
    while (data && got < length) {
        long n = sc(SYS_read, fd, (long)(data + got), length - got, 0, 0, 0);
        if (n <= 0) { ag_plat_free(data); data = 0; break; }
        got += n;
    }
    sc(SYS_close, fd, 0, 0, 0, 0, 0);
    if (data) *size = (size_t)length;
    return data;
}

int ag_plat_write_file(const char *path, const void *data, size_t size) {
    long fd = sc(SYS_openat, AT_FDCWD, (long)path, O_WRONLY | O_CREAT | O_TRUNC, 0644, 0, 0);
    size_t done = 0;
    if (fd < 0) return -1;
    while (done < size) {
        long n = sc(SYS_write, fd, (long)((const char *)data + done), (long)(size - done), 0, 0, 0);
        if (n <= 0) { sc(SYS_close, fd, 0, 0, 0, 0, 0); return -1; }
        done += (size_t)n;
    }
    sc(SYS_close, fd, 0, 0, 0, 0, 0);
    return 0;
}

void ag_plat_puts(const char *text) {
    size_t n = 0;
    while (text[n]) ++n;
    sc(SYS_write, 1, (long)text, (long)n, 0, 0, 0);
}

void ag_plat_put_int(long value) {
    char buf[24];
    int pos = 23;
    unsigned long mag = value < 0 ? (unsigned long)(-(value + 1)) + 1u : (unsigned long)value;
    buf[pos] = 0;
    do { buf[--pos] = (char)('0' + mag % 10); mag /= 10; } while (mag);
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
    for (;;) sc(SYS_exit, code, 0, 0, 0, 0, 0);
}

long ag_plat_clock_ns(void) {
    long ts[2]; /* struct timespec: tv_sec, tv_nsec (both 64-bit on rv64) */
    if (sc(SYS_clock_gettime, 1 /* CLOCK_MONOTONIC */, (long)ts, 0, 0, 0, 0) != 0) return 0;
    return ts[0] * 1000000000L + ts[1];
}

/* QEMU user mode: no L1D maintenance, only program-order fences. */
void ag_plat_ame_sync_in(const void *ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    __asm__ volatile("fence rw,rw" ::: "memory");
}

void ag_plat_ame_sync_out(const void *ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    __asm__ volatile("fence rw,rw" ::: "memory");
}

void ag_plat_ame_release(void) {}
void *ag_plat_ame_addr(const void *ptr) { return (void *)ptr; }

int main(int argc, char **argv);

__attribute__((used)) void ag_start_c(long argc, char **argv) {
    ag_plat_exit(main((int)argc, argv));
}

/* Linux ABI: at _start sp points at argc, followed by argv[]. gp must be
 * set before any global is touched. */
asm(".global _start\n"
    "_start:\n"
    ".option push\n.option norelax\n"
    "  la gp, __global_pointer$\n"
    ".option pop\n"
    "  ld a0, 0(sp)\n"
    "  addi a1, sp, 8\n"
    "  call ag_start_c\n");
