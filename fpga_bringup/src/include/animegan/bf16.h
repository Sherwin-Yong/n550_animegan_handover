#ifndef ANIMEGAN_BF16_H
#define ANIMEGAN_BF16_H
#include <stdint.h>

/* bf16 <-> fp32 as bit operations (no zvfbfmin dependence): bf16 is the upper
 * half of the fp32 pattern; fp32 -> bf16 rounds to nearest even, which is what
 * onnxruntime/numpy do and what the AME's msce32 results get rounded with
 * when written back as activations. NaN is not preserved specially. */
typedef union { float f; uint32_t u; } ag_f32_bits;

static inline float ag_bf16_to_f32(uint16_t h) {
    ag_f32_bits b;
    b.u = (uint32_t)h << 16;
    return b.f;
}

static inline uint16_t ag_f32_to_bf16(float f) {
    ag_f32_bits b;
    b.f = f;
    return (uint16_t)((b.u + 0x7FFFu + ((b.u >> 16) & 1u)) >> 16);
}

#endif
