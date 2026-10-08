#ifndef ANIMEGAN_BOARD_LOG_H
#define ANIMEGAN_BOARD_LOG_H
#include <stdint.h>

/* Board program progress: the current stage and node live in DTCM (src/bsp/trap.c)
 * and are updated in every build, so a trap report and the mailbox (+0x78 stage,
 * +0x7C node) show where the program was. */
enum {
    AG_STAGE_BOOT = 1,      /* reset to magic check */
    AG_STAGE_SELFTEST,      /* CLP self-test */
    AG_STAGE_MODEL,         /* model pack, graph io, plan */
    AG_STAGE_ZERO,          /* graph input and arena cleared */
    AG_STAGE_PRE,           /* UINT8 -> bf16 */
    AG_STAGE_FORWARD,       /* node = ag_board_node */
    AG_STAGE_POST,          /* bf16 -> UINT8 */
    AG_STAGE_DONE,          /* main result written, mailbox 0xB0000004 */
    AG_STAGE_EXTRA,         /* second forward without overlap, frame comparison (node = ag_board_node) */
    AG_STAGE_BENCH,         /* micro-benchmarks (tests/board_bench.c) */
    AG_STAGE_TRAPTEST,      /* trap self-test (ebreak) */
    AG_STAGE_LAST = AG_STAGE_TRAPTEST
};
extern volatile uint32_t ag_board_stage, ag_board_node;
/* also in DTCM, for the trap report: the running node's opcode, the arena and conv
 * scratch sizes (CLP offsets of a faulting address are named after them), and the
 * trap self-test flag (1: the next breakpoint is the self-test; 2: it was handled) */
extern volatile uint32_t ag_board_opcode, ag_trap_selftest;
extern volatile uint64_t ag_board_arena_bytes, ag_board_scratch_bytes;

/* Progress log on the UART, only in debug images (BOARD_DEBUG=1, build-board.sh
 * DEBUG=1): BOARD_LOG(expr, expr, ...) evaluates the comma-separated calls; with
 * BOARD_DEBUG=0 it expands to nothing and its arguments are not evaluated.
 * Banner, magic, self-test, errors, traps and the final cycle counts are always
 * printed. */
#ifndef BOARD_DEBUG
#define BOARD_DEBUG 0
#endif
#if BOARD_DEBUG
#define BOARD_LOG(...) ((void)(__VA_ARGS__))
#else
#define BOARD_LOG(...) ((void)0)
#endif

#endif
