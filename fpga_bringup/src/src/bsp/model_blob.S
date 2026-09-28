/* Embeds the model pack (.agp) into the image so the board needs no file
 * system or host loader: MODEL_PATH is passed by the build script. */
    .section .rodata
    .balign 64
    .global ag_model_blob_start
ag_model_blob_start:
    .incbin MODEL_PATH
    .global ag_model_blob_end
ag_model_blob_end:
    .byte 0
