#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 */
#include <animegan/platform.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void *ag_plat_read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    long length;
    void *data;
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    data = malloc((size_t)length ? (size_t)length : 1);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return 0;
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

int ag_plat_write_file(const char *path, const void *data, size_t size) {
    FILE *file = fopen(path, "wb");
    int ok;
    if (!file) return -1;
    ok = fwrite(data, 1, size, file) == size;
    fclose(file);
    return ok ? 0 : -1;
}

/* 64-byte aligned, zero-filled (docs/memory-plan.md: arena and scratch alignment) */
void *ag_plat_alloc(size_t bytes) {
    void *block = 0;
    size_t size = bytes ? bytes : 1;
    if (posix_memalign(&block, 64, size) != 0) return 0;
    memset(block, 0, size);
    return block;
}
void ag_plat_free(void *ptr) { free(ptr); }
void ag_plat_puts(const char *text) { fputs(text, stdout); fflush(stdout); }
void ag_plat_put_int(long value) { printf("%ld", value); fflush(stdout); }
long ag_plat_atol(const char *text) { return strtol(text, 0, 10); }
void ag_plat_exit(int code) { exit(code); }
long ag_plat_clock_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}
