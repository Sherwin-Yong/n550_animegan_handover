/* Bare-metal entry for QEMU virt and the S2C board. Adapted from the
 * reference bring-up project's start.S: stack, BSS, ITCM/DTCM set-up,
 * FS/VS/MS enable, mtvec, then ag_board_main().
 * ITCM (0x04000000, executable) and DTCM (0x04020000, data) are not reachable
 * over PCIe: the .itcm image (trap entry and src/bsp/trap.c) is copied from its
 * load address in DDR, .dtcm (trap context, trap stack, stage/node) is zeroed,
 * and fence.i orders those stores before the first fetch from ITCM. */
#include "board.h"

#if BOARD_UART_32BIT
#define UART_ST sw
#define UART_LD lw
#else
#define UART_ST sb
#define UART_LD lbu
#endif
#define UART_OFF(idx) ((idx) << BOARD_UART_REG_SHIFT)

    .section .text.init, "ax"
    .global _start
_start:
    fence.i                       /* stale I-cache lines after re-loading an image */
    csrr    t0, mhartid
    bnez    t0, park              /* hart 0 only */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop
    la      sp, _stack_top
    la      t0, _bss_start
    la      t1, _bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  la      t0, _itcm_lma         /* ITCM <- .itcm load image */
    la      t1, _itcm_start
    la      t2, _itcm_end
3:  bgeu    t1, t2, 4f
    ld      t3, 0(t0)
    sd      t3, 0(t1)
    addi    t0, t0, 8
    addi    t1, t1, 8
    j       3b
4:  la      t0, _dtcm_start       /* DTCM <- 0 */
    la      t1, _dtcm_end
5:  bgeu    t0, t1, 6f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       5b
6:  fence.i                       /* ITCM stores before fetching from ITCM */
    li      t0, (1 << 13)         /* mstatus.FS = Initial: float */
    csrs    mstatus, t0
    li      t0, (1 << 9)          /* mstatus.VS = Initial: vector */
    csrs    mstatus, t0
    li      t0, (1 << 29)         /* mstatus.MS = Initial: matrix (AME) */
    csrs    mstatus, t0
    la      t0, trap_entry
    csrw    mtvec, t0
    call    ag_board_main
park:
    wfi
    j       park

/* ---- trap entry (ITCM). ag_trap_frame (DTCM, src/bsp/trap.c): x[0..31] at
 * 8*i, mcause 256, mepc 264, mtval 272, mstatus 280. ag_trap_dispatch returns 1
 * to resume (mepc and registers restored from the frame, mret: an illegal
 * instruction retry, or past the ebreak of the trap self-test) or 0 to halt.
 * A trap while the first is being handled (ag_trap_depth 1) keeps the first
 * context and calls ag_trap_nested on a fresh trap stack (depth 2), which
 * prints both and halts; a trap during that (depth 2) prints one line from
 * here without using the stack and halts. ---- */
    .section .itcm, "ax"
    .align 4
    .global trap_entry
trap_entry:
    csrw    mscratch, t0
    la      t0, ag_trap_depth
    lw      t0, 0(t0)
    bnez    t0, trap_nested
    la      t0, ag_trap_frame
    sd      x1, 8(t0)
    sd      x2, 16(t0)
    sd      x3, 24(t0)
    sd      x4, 32(t0)
    sd      x6, 48(t0)
    sd      x7, 56(t0)
    sd      x8, 64(t0)
    sd      x9, 72(t0)
    sd      x10, 80(t0)
    sd      x11, 88(t0)
    sd      x12, 96(t0)
    sd      x13, 104(t0)
    sd      x14, 112(t0)
    sd      x15, 120(t0)
    sd      x16, 128(t0)
    sd      x17, 136(t0)
    sd      x18, 144(t0)
    sd      x19, 152(t0)
    sd      x20, 160(t0)
    sd      x21, 168(t0)
    sd      x22, 176(t0)
    sd      x23, 184(t0)
    sd      x24, 192(t0)
    sd      x25, 200(t0)
    sd      x26, 208(t0)
    sd      x27, 216(t0)
    sd      x28, 224(t0)
    sd      x29, 232(t0)
    sd      x30, 240(t0)
    sd      x31, 248(t0)
    csrr    t1, mscratch
    sd      t1, 40(t0)            /* x5 = t0 */
    csrr    t1, mcause
    sd      t1, 256(t0)
    csrr    t1, mepc
    sd      t1, 264(t0)
    csrr    t1, mtval
    sd      t1, 272(t0)
    csrr    t1, mstatus
    sd      t1, 280(t0)
    la      t1, ag_trap_depth
    li      t2, 1
    sw      t2, 0(t1)
    la      sp, ag_trap_stack
    li      t1, 4096
    add     sp, sp, t1
    mv      a0, t0
    call    ag_trap_dispatch
    beqz    a0, trap_halt
    la      t0, ag_trap_depth     /* resume: restore mepc and everything else, return */
    sw      zero, 0(t0)
    la      t0, ag_trap_frame
    ld      t1, 264(t0)
    csrw    mepc, t1
    ld      x1, 8(t0)
    ld      x2, 16(t0)
    ld      x3, 24(t0)
    ld      x4, 32(t0)
    ld      x6, 48(t0)
    ld      x7, 56(t0)
    ld      x8, 64(t0)
    ld      x9, 72(t0)
    ld      x10, 80(t0)
    ld      x11, 88(t0)
    ld      x12, 96(t0)
    ld      x13, 104(t0)
    ld      x14, 112(t0)
    ld      x15, 120(t0)
    ld      x16, 128(t0)
    ld      x17, 136(t0)
    ld      x18, 144(t0)
    ld      x19, 152(t0)
    ld      x20, 160(t0)
    ld      x21, 168(t0)
    ld      x22, 176(t0)
    ld      x23, 184(t0)
    ld      x24, 192(t0)
    ld      x25, 200(t0)
    ld      x26, 208(t0)
    ld      x27, 216(t0)
    ld      x28, 224(t0)
    ld      x29, 232(t0)
    ld      x30, 240(t0)
    ld      x31, 248(t0)
    ld      x5, 40(t0)
    mret

trap_nested:                      /* depth 1: report both traps in C on a fresh trap stack */
    la      t0, ag_trap_depth
    lw      t1, 0(t0)
    li      t2, 1
    bne     t1, t2, trap_third
    li      t1, 2
    sw      t1, 0(t0)
    la      sp, ag_trap_stack
    li      t1, 4096
    add     sp, sp, t1
    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval
    call    ag_trap_nested
    j       trap_halt
trap_third:                       /* "NESTED TRAP cause=<hex> epc=<hex> tval=<hex>", no stack */
    li      t6, BOARD_UART_BASE
    la      t5, nested_msg
    call    trap_puts
    csrr    a0, mcause
    call    trap_hex
    la      t5, trap_epc
    call    trap_puts
    csrr    a0, mepc
    call    trap_hex
    la      t5, trap_tval
    call    trap_puts
    csrr    a0, mtval
    call    trap_hex
    la      t5, trap_nl
    call    trap_puts
trap_halt:
#if BOARD_HAS_POWEROFF
    li      t0, BOARD_POWEROFF_ADDR
    li      t1, 0x3333            /* sifive_test FAIL */
    sw      t1, 0(t0)
#endif
7:  wfi
    j       7b

/* t6 = UART base; clobbers t0-t4 and a1 */
trap_putc:                        /* a1 = byte */
8:  UART_LD t0, UART_OFF(UART_LSR)(t6)
    andi    t0, t0, UART_LSR_THRE
    beqz    t0, 8b
    UART_ST a1, UART_OFF(UART_THR)(t6)
    ret
trap_puts:                        /* t5 = string */
    mv      t4, ra
9:  lbu     a1, 0(t5)
    beqz    a1, 10f
    call    trap_putc
    addi    t5, t5, 1
    j       9b
10: mv      ra, t4
    ret
trap_hex:                         /* a0 = value, 16 nibbles */
    mv      t4, ra
    li      t3, 60
11: srl     t1, a0, t3
    andi    t1, t1, 15
    addi    a1, t1, '0'
    li      t2, 10
    blt     t1, t2, 12f
    addi    a1, t1, 'a' - 10
12: call    trap_putc
    addi    t3, t3, -4
    bgez    t3, 11b
    mv      ra, t4
    ret

    .section .itcm.rodata, "a"
nested_msg: .asciz "\r\nTRAP WHILE REPORTING A NESTED TRAP cause=0x"
trap_epc:   .asciz " epc=0x"
trap_tval:  .asciz " tval=0x"
trap_nl:    .asciz "\r\n"
