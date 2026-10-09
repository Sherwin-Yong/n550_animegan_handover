#include <stdint.h>
#include <animegan/board_log.h>
#include "bsp/board.h"

/* Exception service. start.S (trap_entry, ITCM) saves x1..x31 and
 * mcause/mepc/mtval/mstatus into ag_trap_frame (DTCM) and calls
 * ag_trap_dispatch on the DTCM trap stack; a return value of 1 restores mepc
 * and the registers from the frame (mret), 0 halts. A trap inside the handler
 * calls ag_trap_nested on a fresh trap stack; a third level prints one line
 * from start.S and halts. The code, its strings and tables are in ITCM, its
 * data in DTCM; it calls no DDR function or libc and addresses the mailbox and
 * the UART as constants. Every string is a named ITCM array (string literals
 * would land in DDR .rodata); lists are NUL-separated.
 *   mcause 2 illegal instruction: mtval and the instruction at mepc, then
 *     fence.i and retry, at most 3 times for the same mepc.
 *   1/5/7 access fault: the address, its region (docs/n550_mem_map.md) and,
 *     in the CLP window, its slot and offset (arena offsets map to tensors with
 *     tests/runtime_test --plan).
 *   0/4/6 misaligned: the address and its alignment.
 *   3 breakpoint: with ag_trap_selftest = 1 (the board program's trap
 *     self-test) a line and a return past the ebreak; otherwise a hint.
 *   11 ecall: a hint. Other causes and interrupts: the raw mcause.
 * Report order: one summary line (exception, mcause, mepc, mtval, stage, node,
 * opcode); the mailbox (trap fields, stage/node, status 0xB00000FF, written
 * back); then the full context: hint, region, CSRs, mcycle, vl/vtype/vstart
 * (mstatus.VS on), mtilem/n/k (mstatus.MS on), the instruction at mepc (program
 * image or ITCM), the 31 registers and 32 doublewords from sp (stack only).
 * AG_TRAP_HOST_TEST builds the same code on the host for tests/trap_test.c:
 * the UART, the CSRs, the mailbox, the program image and the stack are the
 * test's. */

#ifdef AG_TRAP_HOST_TEST
#define ITCM __attribute__((noinline))
#define ITCM_RO
#define DTCM
void trap_test_putc(int c);
uint64_t trap_test_csr(int id);
void trap_test_mbox_sync(void);
extern uint8_t trap_test_mbox[4096], trap_test_code[64];
extern uint64_t trap_test_stack[64];
#define MBOX_BASE ((uintptr_t)trap_test_mbox)
#define IN_CODE(a) ((a) - (uintptr_t)trap_test_code < sizeof(trap_test_code) - 3)
#define STACK_LO ((uintptr_t)trap_test_stack)
#define STACK_HI ((uintptr_t)(trap_test_stack + 64))
#else
#define ITCM __attribute__((section(".itcm"), noinline))
#define ITCM_RO __attribute__((section(".itcm.rodata")))
#define DTCM __attribute__((section(".dtcm")))
extern char _stack_bottom[], _stack_top[];
#define MBOX_BASE ((uintptr_t)BOARD_MBOX_ADDR)
#define IN_CODE(a) (region_of(a) == 8 || region_of(a) == 3)   /* program image or ITCM */
#define STACK_LO ((uintptr_t)_stack_bottom)
#define STACK_HI ((uintptr_t)_stack_top)
#endif

typedef struct { uint64_t x[32], mcause, mepc, mtval, mstatus; } ag_trap_frame_t;   /* offsets used by start.S */
typedef char ag_trap_frame_check[sizeof(ag_trap_frame_t) == 288 ? 1 : -1];

DTCM ag_trap_frame_t ag_trap_frame;
DTCM uint32_t ag_trap_depth;
DTCM __attribute__((aligned(16))) uint64_t ag_trap_stack[512];   /* 4 KB, grows down from the end */
DTCM volatile uint32_t ag_board_stage, ag_board_node, ag_board_opcode, ag_trap_selftest;
DTCM volatile uint64_t ag_board_arena_bytes, ag_board_scratch_bytes;
DTCM static uint64_t retry_epc;
DTCM static uint32_t retry_count;

static const char s_banner[] ITCM_RO = "\n*** TRAP: ";
static const char s_stars[] ITCM_RO = " ***\n";
static const char s_hint[] ITCM_RO = "hint: ";
static const char s_mcause[] ITCM_RO = " mcause=";
static const char s_mepc[] ITCM_RO = " mepc=";
static const char s_mtval[] ITCM_RO = " mtval=";
static const char s_mstatus[] ITCM_RO = " mstatus=";
static const char s_stage[] ITCM_RO = " stage=";
static const char s_node[] ITCM_RO = " node=";
static const char s_opcode[] ITCM_RO = " opcode=";
static const char s_csrs[] ITCM_RO = "csr:";
static const char s_mcycle[] ITCM_RO = " mcycle=";
static const char s_vl[] ITCM_RO = "vector: vl=";
static const char s_vtype[] ITCM_RO = " vtype=";
static const char s_vstart[] ITCM_RO = " vstart=";
static const char s_voff[] ITCM_RO = "vector: off (mstatus.VS=0)\n";
static const char s_mtilem[] ITCM_RO = "matrix: mtilem=";
static const char s_mtilen[] ITCM_RO = " mtilen=";
static const char s_mtilek[] ITCM_RO = " mtilek=";
static const char s_moff[] ITCM_RO = "matrix: off (mstatus.MS=0)\n";
static const char s_insn_at[] ITCM_RO = "instruction at mepc: ";
static const char s_eq[] ITCM_RO = "=";
static const char s_sp[] ITCM_RO = " ";
static const char s_colon[] ITCM_RO = ": ";
static const char s_plus[] ITCM_RO = " +";
static const char s_nl[] ITCM_RO = "\n";
static const char s_region[] ITCM_RO = "address region: ";
static const char s_align[] ITCM_RO = ", address mod 8 = ";
static const char s_clp[] ITCM_RO = "CLP offset ";
static const char s_arena_hint[] ITCM_RO = " (arena offset: tests/runtime_test --plan names the tensor)";
static const char s_regs[] ITCM_RO = "registers:\n";
static const char s_stack[] ITCM_RO = "stack from sp (doublewords; program addresses are return addresses for addr2line):\n";
static const char s_nostack[] ITCM_RO = "stack: sp outside the stack, not dumped\n";
static const char s_illegal[] ITCM_RO = "\nillegal instruction mepc=";
static const char s_insn[] ITCM_RO = " [mepc]=";
static const char s_retry[] ITCM_RO = ": fence.i and retry ";
static const char s_of3[] ITCM_RO = "/3\n";
static const char s_giveup[] ITCM_RO = ": retries exhausted\n";
static const char s_selftest[] ITCM_RO = "trap self-test ok: breakpoint at mepc=";
static const char s_resume[] ITCM_RO = ", resuming after it\n";
static const char s_nested[] ITCM_RO = "\n*** NESTED TRAP (inside the trap handler): mcause=";
static const char s_first[] ITCM_RO = "\nfirst trap:";
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
static const char clp_slots[] ITCM_RO =
    "input frame\0result frame\0graph input tensor\0graph output tensor slot\0conv scratch\0arena\0past the arena";
static const char stage_names[] ITCM_RO = "?\0boot\0clp self-test\0model/plan\0zero\0pre\0forward\0post\0done\0"
    "extra: second forward / frame comparison (main result already written)\0extra: micro-benchmarks (main result already written)\0"
    "extra: trap self-test (main result already written)";
static const char reg_names[] ITCM_RO =
    "zero\0ra\0sp\0gp\0tp\0t0\0t1\0t2\0s0\0s1\0a0\0a1\0a2\0a3\0a4\0a5\0a6\0a7\0"
    "s2\0s3\0s4\0s5\0s6\0s7\0s8\0s9\0s10\0s11\0t3\0t4\0t5\0t6";

#ifdef AG_TRAP_HOST_TEST
#define tputc trap_test_putc
#else
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
#endif
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

enum { CSR_MCYCLE, CSR_VL, CSR_VTYPE, CSR_VSTART, CSR_MTILEM, CSR_MTILEN, CSR_MTILEK };
ITCM static uint64_t csr(int id) {
#ifdef AG_TRAP_HOST_TEST
    return trap_test_csr(id);
#else
    uint64_t v = 0;
    switch (id) {
    case CSR_MCYCLE: __asm__ volatile("csrr %0, mcycle" : "=r"(v)); break;
    case CSR_VL: __asm__ volatile("csrr %0, 0xc20" : "=r"(v)); break;
    case CSR_VTYPE: __asm__ volatile("csrr %0, 0xc21" : "=r"(v)); break;
    case CSR_VSTART: __asm__ volatile("csrr %0, 0x008" : "=r"(v)); break;
    case CSR_MTILEM: __asm__ volatile("csrr %0, 0x803" : "=r"(v)); break;
    case CSR_MTILEN: __asm__ volatile("csrr %0, 0x804" : "=r"(v)); break;
    case CSR_MTILEK: __asm__ volatile("csrr %0, 0x805" : "=r"(v)); break;
    default: break;
    }
    return v;
#endif
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

/* "CLP offset <off>: <slot> +<offset in the slot>" for an address in either CLP view */
ITCM static void clp_line(uint64_t a) {
    uint64_t off, in;
    unsigned slot;
    if (a - BOARD_CLP_BASE < BOARD_CLP_BYTES) off = a - BOARD_CLP_BASE;
    else if (a - BOARD_CLP_DDR < BOARD_CLP_BYTES) off = a - BOARD_CLP_DDR;
    else return;
    if (off < BOARD_CLP_OUTPUT) { slot = 0; in = off; }
    else if (off < BOARD_CLP_GRAPH_IN) { slot = 1; in = off - BOARD_CLP_OUTPUT; }
    else if (off < BOARD_CLP_OUT_TENSOR) { slot = 2; in = off - BOARD_CLP_GRAPH_IN; }
    else if (off < BOARD_CLP_ARENA) { slot = 3; in = off - BOARD_CLP_OUT_TENSOR; }
    else {
        in = off - BOARD_CLP_ARENA;
        slot = in < ag_board_scratch_bytes ? 4 : in < ag_board_arena_bytes ? 5 : 6;
    }
    tputs(s_clp); thex(off); tputs(s_colon); tputs(nth(clp_slots, slot)); tputs(s_plus); thex(in);
    if (slot >= 4) tputs(s_arena_hint);
    tputs(s_nl);
}

/* the 16- or 32-bit instruction at a program-image or ITCM address, 0 elsewhere */
ITCM static uint32_t insn_at(uint64_t pc) {
    uint32_t insn;
    if (!IN_CODE(pc)) return 0;
    insn = *(volatile uint16_t *)(uintptr_t)pc;
    if ((insn & 3) == 3) insn |= (uint32_t)*(volatile uint16_t *)(uintptr_t)(pc + 2) << 16;
    return insn;
}

/* fence (+ l1d_clean_all + fence on the board): mailbox lines to DDR */
ITCM static void mbox_writeback(void) {
#ifdef AG_TRAP_HOST_TEST
    trap_test_mbox_sync();
#else
    __asm__ volatile("fence rw,rw" ::: "memory");
#if BOARD_L1D_ALL
    __asm__ volatile("l1d_clean_all\n\tfence rw,rw" ::: "memory");
#endif
#endif
}

/* trap fields first (they span two cache lines), then the status word, each
 * written back, so the host never sees 0xB00000FF before the fields */
ITCM static void mbox_report(const ag_trap_frame_t *f) {
    volatile uint64_t *t = (volatile uint64_t *)(MBOX_BASE + (BOARD_MBOX_TRAP - BOARD_MBOX_ADDR));
    volatile uint32_t *s = (volatile uint32_t *)(MBOX_BASE + (BOARD_MBOX_STAGE - BOARD_MBOX_ADDR));
    t[0] = f->mcause; t[1] = f->mepc; t[2] = f->mtval; t[3] = f->mstatus; t[4] = f->x[1]; t[5] = f->x[2];
    s[0] = ag_board_stage; s[1] = ag_board_node;
    mbox_writeback();
    *(volatile uint32_t *)MBOX_BASE = BOARD_ST_TRAP;
    mbox_writeback();
}

ITCM static void where(void) {
    uint32_t stage = ag_board_stage;
    tputs(s_stage); tdec(stage); tputs(s_sp); tputs(nth(stage_names, stage <= AG_STAGE_LAST ? stage : 0));
    tputs(s_node); tdec(ag_board_node); tputs(s_opcode); tdec(ag_board_opcode);
}

/* "*** TRAP: <exception> mcause=.. mepc=.. mtval=.. stage=<n> <name> node=<n> opcode=<n> ***" */
ITCM static void summary(const ag_trap_frame_t *f, unsigned idx) {
    tputs(s_banner); tputs(nth(cause_names, idx));
    tputs(s_mcause); thex(f->mcause); tputs(s_mepc); thex(f->mepc); tputs(s_mtval); thex(f->mtval);
    where();
    tputs(s_stars);
}

ITCM static void context(const ag_trap_frame_t *f, unsigned idx) {
    uint64_t sp = f->x[2], a;
    uint32_t insn = insn_at(f->mepc);
    unsigned i;
    tputs(s_hint); tputs(nth(cause_hints, idx)); tputs(s_nl);
    if (idx == 1 || idx == 5 || idx == 7 || idx == 0 || idx == 4 || idx == 6) {
        tputs(s_region); thex(f->mtval); tputs(s_sp); tputs(nth(region_names, region_of(f->mtval)));
        if (idx == 0 || idx == 4 || idx == 6) { tputs(s_align); tdec(f->mtval & 7); }
        tputs(s_nl);
        clp_line(f->mtval);
    }
    tputs(s_csrs); tputs(s_mcause); thex(f->mcause); tputs(s_mepc); thex(f->mepc); tputs(s_mtval); thex(f->mtval);
    tputs(s_mstatus); thex(f->mstatus); tputs(s_mcycle); tdec(csr(CSR_MCYCLE)); tputs(s_nl);
    if ((f->mstatus >> 9) & 3) {
        tputs(s_vl); tdec(csr(CSR_VL)); tputs(s_vtype); thex(csr(CSR_VTYPE)); tputs(s_vstart); tdec(csr(CSR_VSTART)); tputs(s_nl);
    } else tputs(s_voff);
    if ((f->mstatus >> 29) & 3) {
        tputs(s_mtilem); tdec(csr(CSR_MTILEM)); tputs(s_mtilen); tdec(csr(CSR_MTILEN)); tputs(s_mtilek); tdec(csr(CSR_MTILEK)); tputs(s_nl);
    } else tputs(s_moff);
    if (insn) { tputs(s_insn_at); thex(insn); tputs(s_nl); }
    tputs(s_regs);
    for (i = 1; i < 32; ++i) {
        tputs(nth(reg_names, i)); tputs(s_eq); thex(f->x[i]); tputs(i % 4 == 3 ? s_nl : s_sp);
    }
    tputs(s_nl);
    if (sp >= STACK_LO && sp < STACK_HI && !(sp & 7)) {
        tputs(s_stack);
        for (i = 0, a = sp; i < 32 && a < STACK_HI; ++i, a += 8) {
            if (i % 4 == 0) { thex(a); tputs(s_colon); }
            thex(*(volatile uint64_t *)(uintptr_t)a); tputs(i % 4 == 3 ? s_nl : s_sp);
        }
        if (i % 4) tputs(s_nl);
    } else tputs(s_nostack);
    tputs(s_halt);
}

ITCM int ag_trap_dispatch(ag_trap_frame_t *f) {
    uint64_t code = f->mcause & 0xFFF;
    unsigned idx = (f->mcause >> 63) ? 17 : code < 16 ? (unsigned)code : 16;
    if (idx == 3 && ag_trap_selftest == 1) {   /* the board program's trap self-test */
        uint32_t insn = insn_at(f->mepc);
        ag_trap_selftest = 2;
        tputs(s_selftest); thex(f->mepc); tputs(s_resume);
        f->mepc += (insn & 3) == 3 ? 4 : 2;
        return 1;
    }
    if (idx == 2) {
        uint32_t insn = insn_at(f->mepc);
        if (f->mepc != retry_epc) { retry_epc = f->mepc; retry_count = 0; }
        tputs(s_illegal); thex(f->mepc); tputs(s_mtval); thex(f->mtval); tputs(s_insn); thex(insn);
        if (retry_count < 3) {
            ++retry_count;
            tputs(s_retry); tdec(retry_count); tputs(s_of3);
#ifndef AG_TRAP_HOST_TEST
            __asm__ volatile("fence.i" ::: "memory");
#endif
            return 1;
        }
        tputs(s_giveup);
    }
    summary(f, idx);
    mbox_report(f);
    context(f, idx);
    return 0;
}

/* A trap inside ag_trap_dispatch (start.S, on a fresh trap stack): this trap's
 * cause, epc and tval, the first trap's from the frame and where the program
 * was; start.S halts afterwards. */
ITCM void ag_trap_nested(uint64_t cause, uint64_t epc, uint64_t tval) {
    tputs(s_nested); thex(cause); tputs(s_mepc); thex(epc); tputs(s_mtval); thex(tval);
    tputs(s_first); tputs(s_mcause); thex(ag_trap_frame.mcause); tputs(s_mepc); thex(ag_trap_frame.mepc); tputs(s_mtval); thex(ag_trap_frame.mtval);
    tputs(s_nl); where(); tputs(s_nl); tputs(s_halt);
}
