#ifndef ANIMEGAN_BSP_BOARD_H
#define ANIMEGAN_BSP_BOARD_H
/* Board constants for the bare-metal build. Two targets:
 *   -DBOARD_QEMU_VIRT  qemu-system-riscv64 -M virt (16550 at 0x10000000, 8-bit
 *                      registers, no baud setup, sifive_test poweroff)
 *   -DBOARD_S2C        N550+AMU FPGA prototype: DW_apb_uart 16550-compatible at
 *                      0x20100000, 32-bit registers 4 bytes apart, baud rate set
 *                      by the program, CPU/AME 40 MHz; address map in
 *                      docs/n550_mem_map.md.
 * Values for the S2C board come from the reference bring-up project
 * (n550_qwen3_bringup_handover/fpga_bringup/src/src/bsp/board.h, BUILD.md).
 * Included from start.S too: keep every constant a plain literal. */

#if defined(BOARD_QEMU_VIRT)
#define BOARD_NAME "QEMU virt"
#define BOARD_UART_BASE 0x10000000
#define BOARD_UART_32BIT 0
#define BOARD_UART_REG_SHIFT 0
#define BOARD_UART_NEEDS_INIT 0
#define BOARD_HAS_POWEROFF 1
#define BOARD_POWEROFF_ADDR 0x100000
#define BOARD_CPU_HZ 0            /* mcycle is not wall time under emulation */
#define BOARD_L1D_ALL 0           /* l1d_clean_all is illegal under QEMU: fence only */
#define BOARD_CLP_BASE 0xF0000000 /* no CLP window: the RVV view is the DDR view (needs -m 2G) */
#elif defined(BOARD_S2C)
#define BOARD_NAME "S2C N550+AMU"
#define BOARD_UART_BASE 0x20100000
#define BOARD_UART_32BIT 1
#define BOARD_UART_REG_SHIFT 2
#define BOARD_UART_NEEDS_INIT 1
#define BOARD_HAS_POWEROFF 0
#define BOARD_CPU_HZ 40000000
#define BOARD_L1D_ALL 1           /* xdcache l1d_clean_all writes the mailbox back */
#define BOARD_CLP_BASE 0x10000000 /* CLP window: RVV only, uncached, NoC -> BOARD_CLP_DDR */
/* UART input clock 10 MHz (D34); overridable at build time. */
#ifndef BOARD_UART_CLK_HZ
#define BOARD_UART_CLK_HZ 10000000
#endif
#define BOARD_UART_HAS_DLF 1
#define BOARD_UART_DLF_IDX 48
#define BOARD_UART_DLF_BITS 4
#else
#error "define BOARD_QEMU_VIRT or BOARD_S2C"
#endif

#ifndef BOARD_UART_BAUD
#define BOARD_UART_BAUD 115200
#endif
#define BOARD_DRAM_BASE 0x80000000
/* Heap for the bump allocator (scalar-only data: tensor index arrays); the
 * program image (text/data/bss/stack) must end below the mailbox. */
#define BOARD_HEAP_BASE 0x88000000
#define BOARD_HEAP_BYTES 0x20000000   /* 512 MB */
#define BOARD_CACHE_LINE 64
/* Tensor data lives in DDR BOARD_CLP_DDR.. and is touched only by RVV through
 * the RVV view BOARD_CLP_BASE.. (same offsets) and by the AMU through the DDR
 * view; the host reads and writes the DDR view over PCIe. Offsets: */
#define BOARD_CLP_DDR 0xF0000000
#define BOARD_CLP_BYTES 0x10000000    /* 256 MB */
#define BOARD_CLP_INPUT 0x0           /* input frame */
#define BOARD_CLP_OUTPUT 0x100000     /* result frame */
#define BOARD_CLP_GRAPH_IN 0x200000   /* dense bf16 graph input tensor */
#define BOARD_CLP_OUT_TENSOR 0x400000 /* dense bf16 graph output tensor, 2 MB slot at a fixed host address */
#define BOARD_CLP_ARENA 0x600000      /* activation arena, conv scratch at offset 0 */
/* Back-door (PCIe) hand-over, tests/board_backdoor.c: while the CPU is held in reset
 * the host writes the program (raw binary at the DRAM base), BOARD_HOST_MAGIC at
 * BOARD_MBOX_MAGIC and one frame at BOARD_INPUT_ADDR, then releases reset, polls the
 * mailbox status and reads the result frame at BOARD_OUTPUT_ADDR and the graph output
 * tensor at BOARD_OUT_TENSOR_ADDR (DDR view). Frames are 512x512 RGB UINT8. */
#define BOARD_MBOX_ADDR 0x87F00000    /* 4 KB mailbox (status, cycles, trap, stage) */
#define BOARD_MBOX_MAGIC 0x87F00080   /* host-written word, echoed at +0x84 */
#define BOARD_HOST_MAGIC 0x5A5AC3C3
#define BOARD_INPUT_ADDR 0xF0000000   /* BOARD_CLP_DDR + BOARD_CLP_INPUT */
#define BOARD_OUTPUT_ADDR 0xF0100000  /* BOARD_CLP_DDR + BOARD_CLP_OUTPUT */
#define BOARD_FRAME_BYTES 786432
#define BOARD_OUT_TENSOR_ADDR 0xF0400000 /* BOARD_CLP_DDR + BOARD_CLP_OUT_TENSOR, repeated at mailbox +0x38 */
#define BOARD_OUT_TENSOR_BYTES 1572864   /* bf16 [1,512,512,3], repeated at mailbox +0x40 */
#define BOARD_MBOX_TRAP 0x87F00048    /* trap: mcause, mepc, mtval, mstatus, ra, sp (u64) */
#define BOARD_MBOX_STAGE 0x87F00078   /* u32 stage, u32 node (include/animegan/board_log.h) */
#define BOARD_ST_TRAP 0xB00000FF      /* mailbox status after a trap */

/* 16550 register indices (byte offset = index << BOARD_UART_REG_SHIFT) */
#define UART_RBR 0
#define UART_THR 0
#define UART_DLL 0
#define UART_IER 1
#define UART_DLM 1
#define UART_FCR 2
#define UART_LCR 3
#define UART_LSR 5
#define UART_LSR_DR 0x01
#define UART_LSR_THRE 0x20

#endif
