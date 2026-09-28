#ifndef ANIMEGAN_PLATFORM_H
#define ANIMEGAN_PLATFORM_H
#include <stddef.h>

/* Minimal host services used by the test drivers so that the same driver
 * source builds for the POSIX host and for freestanding RV64 Linux user mode. */
void *ag_plat_read_file(const char *path, size_t *size);
int ag_plat_write_file(const char *path, const void *data, size_t size);
void *ag_plat_alloc(size_t bytes); /* zero-filled */
void ag_plat_free(void *ptr);
void ag_plat_puts(const char *text);
void ag_plat_put_int(long value);
long ag_plat_atol(const char *text);
void ag_plat_exit(int code);
/* Monotonic time in nanoseconds for per-node timing (--time); the board layer
 * maps it onto its cycle counter. QEMU values are functional only. */
long ag_plat_clock_ns(void);
/* Memory ordering at the AME boundary: called before the matrix unit reads
 * or writes [ptr, ptr+bytes) and after it has written it (fence rw,rw: every
 * buffer the matrix unit shares is uncached on the board). */
void ag_plat_ame_sync_in(const void *ptr, size_t bytes);
void ag_plat_ame_sync_out(const void *ptr, size_t bytes);
/* Address the matrix unit uses for a buffer the runtime addresses as ptr: the
 * board maps its RVV window (tensor data) to the DDR view; other addresses and
 * QEMU user mode are unchanged. */
void *ag_plat_ame_addr(const void *ptr);
/* End of a matrix-unit section. User mode cannot touch mstatus.MS, so QEMU
 * user mode does nothing; the board layer marks the context Clean. */
void ag_plat_ame_release(void);

#endif
