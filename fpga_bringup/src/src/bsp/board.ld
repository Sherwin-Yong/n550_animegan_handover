/* Bare-metal image layout: text/rodata (with the embedded model pack)
 * /data, then the .itcm load image, then bss/stack from DRAM base; the image
 * must end below the back-door mailbox BOARD_MBOX_ADDR (board.h, checked by
 * scripts/build-board.sh). .itcm (trap entry and handler) runs from ITCM and
 * .dtcm (trap context, trap stack, stage/node) lives in DTCM; start.S copies
 * .itcm from _itcm_lma and zeroes .dtcm. */
OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
    RAM (rwxa) : ORIGIN = 0x80000000, LENGTH = 128M
    ITCM (rx) : ORIGIN = 0x04000000, LENGTH = 128K
    DTCM (rw) : ORIGIN = 0x04020000, LENGTH = 128K
}
_stack_size = 256K;
SECTIONS
{
    .text : { KEEP(*(.text.init)) *(.text .text.*) } > RAM
    .rodata : { . = ALIGN(64); *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
    .data : {
        . = ALIGN(8); *(.data .data.*)
        . = ALIGN(8); __global_pointer$ = . + 0x800; *(.sdata .sdata.*)
        . = ALIGN(8);   /* the .itcm load address follows: start.S copies it with ld */
    } > RAM
    .itcm : { . = ALIGN(8); _itcm_start = .; KEEP(*(.itcm .itcm.*)) . = ALIGN(8); _itcm_end = .; } > ITCM AT > RAM
    _itcm_lma = LOADADDR(.itcm);
    .bss (NOLOAD) : {
        . = ALIGN(8); _bss_start = .;
        *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON)
        . = ALIGN(8); _bss_end = .;
    } > RAM
    .stack (NOLOAD) : { . = ALIGN(16); _stack_bottom = .; . += _stack_size; _stack_top = .; } > RAM
    _end = .;
    .dtcm (NOLOAD) : { . = ALIGN(16); _dtcm_start = .; *(.dtcm .dtcm.*) . = ALIGN(8); _dtcm_end = .; } > DTCM
    ASSERT(_bss_start >= _itcm_lma + SIZEOF(.itcm), ".bss overlaps the .itcm load image")
    /* N550 has no misaligned access: the ld/sd loops of start.S need 8-byte bounds */
    ASSERT(_itcm_lma % 8 == 0 && _itcm_start % 8 == 0 && _itcm_end % 8 == 0, ".itcm copy bounds not 8-byte aligned")
    ASSERT(_bss_start % 8 == 0 && _bss_end % 8 == 0, ".bss bounds not 8-byte aligned")
    ASSERT(_dtcm_start % 8 == 0 && _dtcm_end % 8 == 0, ".dtcm bounds not 8-byte aligned")
    ASSERT(_stack_top % 16 == 0, "stack top not 16-byte aligned")
    /DISCARD/ : { *(.comment) *(.note.*) *(.eh_frame) }
}
