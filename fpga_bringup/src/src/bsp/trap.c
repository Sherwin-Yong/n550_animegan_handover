#include <stdint.h>
#include <animegan/board_log.h>
#include "bsp/board.h"

/* Exception service. start.S (trap_entry, ITCM) saves x1..x31 and
 * mcause/mepc/mtval/mstatus into ag_trap_frame (DTCM) and calls
 * ag_trap_dispatch on the DTCM trap stack; a return value of 1 restores the
 * registers and retries the instruction (mret), 0 halts. The code, its strings
 * and tables are in ITCM, its data in DTCM; it calls no DDR function or libc and
 * addresses the mailbox and the UART as constants. Every string is a named ITCM
 * array (string literals would land in DDR .rodata); lists are NUL-separated.
 *   mcause 2 illegal instruction: mtval and the instruction bytes at mepc, then
 *     fence.i and retry, at most 3 times for the same mepc.
 *   1/5/7 access fault: the address and its region (docs/n550_mem_map.md).
 *   0/4/6 misaligned: the address and its alignment.
 *   3 breakpoint, 11 ecall: a hint. Other causes and interrupts: the raw mcause.
 * The report writes the mailbox trap fields, stage/node and status 0xB00000FF,
 * writes L1D back, then prints the exception, hint, CSRs, stage/node and all
 * registers on the UART. */

#define ITCM __attribute__((section(".itcm"), noinline))
#define ITCM_RO __attribute__((section(".itcm.rodata")))
#define DTCM __attribute__((section(".dtcm")))

typedef struct { uint64_t x[32], mcause, mepc, mtval, mstatus; } ag_trap_frame_t;   /* offsets used by start.S */
typedef char ag_trap_frame_check[sizeof(ag_trap_frame_t) == 288 ? 1 : -1];

DTCM ag_trap_frame_t ag_trap_frame;
DTCM uint32_t ag_trap_depth;
DTCM __attribute__((aligned(16))) uint64_t ag_trap_stack[512];   /* 4 KB, grows down from the end */
DTCM volatile uint32_t ag_board_stage, ag_board_node;
DTCM static uint64_t retry_epc;
DTCM static uint32_t retry_count;

static const char s_banner[] ITCM_RO = "\n*** TRAP: ";
static const char s_stars[] ITCM_RO = " ***\n";
static const char s_hint[] ITCM_RO = "hint: ";
static const char s_mcause[] ITCM_RO = "mcause=";
static const char s_mepc[] ITCM_RO = " mepc=";
static const char s_mtval[] ITCM_RO = " mtval=";
static const char s_mstatus[] ITCM_RO = " mstatus=";
static const char s_stage[] ITCM_RO = "\nstage=";
static const char s_node[] ITCM_RO = " node=";
static const char s_eq[] ITCM_RO = "=";
static const char s_sp[] ITCM_RO = " ";
static const char s_nl[] ITCM_RO = "\n";
static const char s_region[] ITCM_RO = "address region: ";
static const char s_align[] ITCM_RO = ", address mod 8 = ";
static const char s_illegal[] ITCM_RO = "\nillegal instruction mepc=";
static const char s_insn[] ITCM_RO = " [mepc]=";
static const char s_retry[] ITCM_RO = ": fence.i and retry ";
static const char s_of3[] ITCM_RO = "/3\n";
static const char s_giveup[] ITCM_RO = ": retries exhausted\n";
static const char s_halt[] ITCM_RO = "halted\n";

static const char cause_names[] ITCM_RO =
    "instruction address misaligned\0instruction access fault\0illegal instruction\0breakpoint\0"
    "load address misaligned\0load access fault\0store/AMO address misaligned\0store/AMO access fault\0"
    "ecall from U-mode\0ecall from S-mode\0reserved exception\0ecall from M-mode\0"
    "instruction page fault\0load page fault\0reserved exception\0store/AMO page fault\0"
    "exception (cause >= 16)\0interrupt (interrupts are not enabled)";
static const char cause_hints[] ITCM_RO =
    "jump or return to an address that is not 2-byte aligned (corrupted ra or function pointer)\0"
    "fetch from a non-executable or unmapped region (DTCM, CLP, tensor data): corrupted ra/pointer\0"
    "unknown or disabled instruction; [mepc] shows the bytes in memory (retried after fence.i)\0"
    "ebreak reached: debug break or corrupted code\0"
    "load not aligned to its size\0"
    "load from a region the scalar core may not read (CLP window is RVV-only) or unmapped\0"
    "store not aligned to its size\0"
    "store to a region the scalar core may not write (CLP window is RVV-only) or unmapped\0"
    "unexpected ecall\0unexpected ecall\0unexpected cause\0ecall: bare metal has no environment calls\0"
    "no MMU in use\0no MMU in use\0unexpected cause\0no MMU in use\0"
    "unexpected cause\0unexpected interrupt";
static const char region_names[] ITCM_RO =
    "debug\0CLINT\0APLIC\0ITCM\0DTCM\0CLP window (RVV only: scalar access faults)\0peripheral\0system port\0"
    "program image\0mailbox\0heap (scalar data)\0tensor data (CLP DDR view: AMU and host)\0DDR (unused)\0unmapped";
static const char stage_names[] ITCM_RO = "?\0boot\0clp self-test\0model/plan\0zero\0pre\0forward\0post\0done";
static const char reg_names[] ITCM_RO =
    "zero\0ra\0sp\0gp\0tp\0t0\0t1\0t2\0s0\0s1\0a0\0a1\0a2\0a3\0a4\0a5\0a6\0a7\0"
    "s2\0s3\0s4\0s5\0s6\0s7\0s8\0s9\0s10\0s11\0t3\0t4\0t5\0t6";

#define UART_REG(idx) ((uintptr_t)BOARD_UART_BASE + ((uintptr_t)(idx) << BOARD_UART_REG_SHIFT))
#if BOARD_UART_32BIT
#define UART_RD(idx) (*(volatile uint32_t *)UART_REG(idx))
#define UART_WR(idx, v) (*(volatile uint32_t *)UART_REG(idx) = (uint32_t)(v))
#else
#define UART_RD(idx) (*(volatile uint8_t *)UART_REG(idx))
#define UART_WR(idx, v) (*(volatile uint8_t *)UART_REG(idx) = (uint8_t)(v))
#endif

ITCM static void tputc(int c) {
    long spin = 0;
    while (!(UART_RD(UART_LSR) & UART_LSR_THRE) && ++spin < 1000000) {}
    UART_WR(UART_THR, c & 0xFF);
}
ITCM static void tputs(const char *s) {
    for (; *s; ++s) {
        if (*s == '\n') tputc('\r');
        tputc(*s);
    }
}
ITCM static void thex(uint64_t v) {
    int sh;
    tputc('0'); tputc('x');
    for (sh = 60; sh >= 0; sh -= 4) { int d = (int)(v >> sh) & 15; tputc(d < 10 ? '0' + d : 'a' + d - 10); }
}
ITCM static void tdec(uint64_t v) {
    char buf[21];
    int pos = 20;
    buf[pos] = 0;
    do { buf[--pos] = (char)('0' + v % 10); v /= 10; } while (v);
    tputs(buf + pos);
}
ITCM static const char *nth(const char *list, unsigned i) {
    while (i--) while (*list++) {}
    return list;
}

ITCM static unsigned region_of(uint64_t a) {
    if (a < 0x1000) return 0;
    if (a - 0x02000000u < 0x10000u) return 1;
    if (a - 0x03000000u < 0x10000u) return 2;
    if (a - 0x04000000u < 0x20000u) return 3;
    if (a - 0x04020000u < 0x20000u) return 4;
#if BOARD_CLP_BASE != BOARD_CLP_DDR
    if (a - BOARD_CLP_BASE < BOARD_CLP_BYTES) return 5;
#endif
    if (a - 0x20000000u < 0x20000000u) return 6;
    if (a - 0x40000000u < 0x20000000u) return 7;
    if (a - BOARD_DRAM_BASE < (uint64_t)(BOARD_MBOX_ADDR - BOARD_DRAM_BASE)) return 8;
    if (a - BOARD_MBOX_ADDR < 0x1000u) return 9;
    if (a - BOARD_HEAP_BASE < BOARD_HEAP_BYTES) return 10;
    if (a - BOARD_CLP_DDR < BOARD_CLP_BYTES) return 11;
    if (a - 0x80000000u < 0x80000000u) return 12;
    return 13;
}

/* fence (+ l1d_clean_all + fence on the board): mailbox lines to DDR */
ITCM static void mbox_writeback(void) {
    __asm__ volatile("fence rw,rw" ::: "memory");
#if BOARD_L1D_ALL
    __asm__ volatile("l1d_clean_all\n\tfence rw,rw" ::: "memory");
#endif
}

/* trap fields first (they span two cache lines), then the status word, each
 * written back, so the host never sees 0xB00000FF before the fields */
ITCM static void mbox_report(const ag_trap_frame_t *f) {
    volatile uint64_t *t = (volatile uint64_t *)(uintptr_t)BOARD_MBOX_TRAP;
    volatile uint32_t *s = (volatile uint32_t *)(uintptr_t)BOARD_MBOX_STAGE;
    t[0] = f->mcause; t[1] = f->mepc; t[2] = f->mtval; t[3] = f->mstatus; t[4] = f->x[1]; t[5] = f->x[2];
    s[0] = ag_board_stage; s[1] = ag_board_node;
    mbox_writeback();
    *(volatile uint32_t *)(uintptr_t)BOARD_MBOX_ADDR = BOARD_ST_TRAP;
    mbox_writeback();
}

ITCM int ag_trap_dispatch(ag_trap_frame_t *f) {
    uint64_t code = f->mcause & 0xFFF;
    unsigned idx = (f->mcause >> 63) ? 17 : code < 16 ? (unsigned)code : 16, i;
    if (idx == 2) {
        uint32_t insn = *(volatile uint16_t *)(uintptr_t)f->mepc;
        if ((insn & 3) == 3) insn |= (uint32_t)*(volatile uint16_t *)(uintptr_t)(f->mepc + 2) << 16;
        if (f->mepc != retry_epc) { retry_epc = f->mepc; retry_count = 0; }
        tputs(s_illegal); thex(f->mepc); tputs(s_mtval); thex(f->mtval); tputs(s_insn); thex(insn);
        if (retry_count < 3) {
            ++retry_count;
            tputs(s_retry); tdec(retry_count); tputs(s_of3);
            __asm__ volatile("fence.i" ::: "memory");
            return 1;
        }
        tputs(s_giveup);
    }
    mbox_report(f);
    tputs(s_banner); tputs(nth(cause_names, idx)); tputs(s_stars);
    tputs(s_hint); tputs(nth(cause_hints, idx)); tputs(s_nl);
    if (idx == 1 || idx == 5 || idx == 7 || idx == 0 || idx == 4 || idx == 6) {
        tputs(s_region); thex(f->mtval); tputs(s_sp); tputs(nth(region_names, region_of(f->mtval)));
        if (idx == 0 || idx == 4 || idx == 6) { tputs(s_align); tdec(f->mtval & 7); }
        tputs(s_nl);
    }
    tputs(s_mcause); thex(f->mcause); tputs(s_mepc); thex(f->mepc); tputs(s_mtval); thex(f->mtval); tputs(s_mstatus); thex(f->mstatus);
    tputs(s_stage); tputs(nth(stage_names, ag_board_stage <= AG_STAGE_DONE ? ag_board_stage : 0)); tputs(s_node); tdec(ag_board_node); tputs(s_nl);
    for (i = 1; i < 32; ++i) {
        tputs(nth(reg_names, i)); tputs(s_eq); thex(f->x[i]); tputs(i % 4 == 3 ? s_nl : s_sp);
    }
    tputs(s_nl); tputs(s_halt);
    return 0;
}
