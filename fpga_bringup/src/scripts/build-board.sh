#!/usr/bin/env bash
# Bare-metal back-door frame program: tests/board_backdoor.c + runtime + platform_board
# + bsp, model pack embedded. BOARD=qemu_virt (default) or s2c; BACKEND=ame
# (default) or scalar; DEBUG=0 (default, release) or 1 (per-stage/per-node UART
# log); MODEL=<pack> (default build/animeganv3_bf16.agp).
# Output: $ANIMEGAN_BUILD/board/board_<board>_<backend>[_debug].elf, .bin (raw, loaded
# at the DRAM base by the host) and .txt (back-door load addresses).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
root_dir=$ANIMEGAN_ROOT
board=${BOARD:-qemu_virt}
backend=${BACKEND:-ame}
debug=${DEBUG:-0}
model=${MODEL:-$root_dir/build/animeganv3_bf16.agp}
out_dir=$ANIMEGAN_BUILD/board
mkdir -p "$out_dir"
[[ -f $model ]] || { echo "model pack not found: $model" >&2; exit 1; }
[[ $debug == 0 || $debug == 1 ]] || { echo "DEBUG must be 0 or 1" >&2; exit 2; }
core="$root_dir/src/runtime.c $root_dir/src/executor.c $root_dir/src/planner.c $root_dir/src/conv_bf16.c $root_dir/src/ops_bf16.c
      $root_dir/src/libc_min.c $root_dir/src/platform_board.c
      $root_dir/src/bsp/start.S $root_dir/src/bsp/trap.c $root_dir/src/bsp/model_blob.S"
defs=(-DBOARD_BACKEND_STR="\"$backend\"" -DMODEL_PATH="\"$model\"" -DBOARD_DEBUG=$debug)
case $board in
    qemu_virt) defs+=(-DBOARD_QEMU_VIRT) ;;
    s2c) defs+=(-DBOARD_S2C) ;;
    *) echo "BOARD must be qemu_virt or s2c" >&2; exit 2 ;;
esac
case $backend in
    ame) core="$core $root_dir/src/gemm_ame.c $root_dir/src/conv_ame.c $root_dir/src/ops_rvv.c"; defs+=(-DAG_AME -DAG_RVV -DAG_NO_SCALAR_FALLBACK -DBOARD_BACKEND=AG_BACKEND_AME) ;;
    scalar) defs+=(-DBOARD_BACKEND=AG_BACKEND_SCALAR) ;;
    *) echo "BACKEND must be ame or scalar" >&2; exit 2 ;;
esac
# tensor data sits in the CLP window, which only RVV may access on the board
[[ $board != s2c || $backend == ame ]] || { echo "BOARD=s2c needs BACKEND=ame (scalar code cannot access the CLP window)" >&2; exit 2; }
flags="-march=$ANIMEGAN_MARCH -mabi=lp64d -mcmodel=medany -O2 -std=gnu11 -g3 -gdwarf-4
       -ffp-contract=off -fno-math-errno -fno-tree-vectorize -fno-tree-slp-vectorize
       -ffreestanding -nostdlib -nostartfiles -Wall -Wextra -Werror
       -I$root_dir/include -I$root_dir/src -T $root_dir/src/bsp/board.ld ${BOARD_EXTRA_DEFS:-}"
suffix=$([[ $debug == 1 ]] && echo _debug || true)
elf=$out_dir/board_${board}_${backend}${suffix}.elf
"$CC" $flags "${defs[@]}" $core "$root_dir/tests/board_backdoor.c" -lm -o "$elf"
"${CC%gcc}objcopy" -O binary --gap-fill 0 "$elf" "${elf%.elf}.bin"
addr() { cpp -P "${defs[@]}" -dM "$root_dir/src/bsp/board.h" | awk -v k="$1" '$2==k{print $3}'; }
sym() { "${CC%gcc}nm" "$elf" | awk -v k="$1" '$3==k{print $1}'; }
end=$(sym _end)
(( 0x$end <= $(addr BOARD_MBOX_ADDR) )) || { echo "image ends at 0x$end, past the mailbox $(addr BOARD_MBOX_ADDR)" >&2; exit 1; }
(( 0x$(sym trap_entry) == 0x04000000 && 0x$(sym _dtcm_start) == 0x04020000 )) || { echo "trap_entry / .dtcm not at ITCM / DTCM base" >&2; exit 1; }
{
    echo "back-door load addresses ($board, $backend, DEBUG=$debug)"
    echo "  program  $(basename "${elf%.elf}.bin") -> $(addr BOARD_DRAM_BASE)  ($(stat -c %s "${elf%.elf}.bin") B, image ends 0x$end)"
    echo "  magic    32-bit $(addr BOARD_HOST_MAGIC) -> $(addr BOARD_MBOX_MAGIC)  (a value, not a file)"
    echo "  input    512x512 RGB UINT8 frame ($(addr BOARD_FRAME_BYTES) B) -> $(addr BOARD_INPUT_ADDR)"
    echo "  output   read $(addr BOARD_FRAME_BYTES) B at $(addr BOARD_OUTPUT_ADDR)"
    echo "  mailbox  $(addr BOARD_MBOX_ADDR): status 0xB0000004 done / 0xB00000EE error (code +0x04) / 0xB00000FF trap (+0x48)"
    echo "  tensor   read the bf16 graph output tensor at the mailbox +0x38 address, +0x40 bytes"
    echo "  order: hold reset; write program, 0 to the status word $(addr BOARD_MBOX_ADDR), magic, input -> release reset -> poll mailbox status -> read output, output tensor and mailbox"
} > "${elf%.elf}.txt"
"${CC%gcc}size" "$elf"
cat "${elf%.elf}.txt"
echo "board image build PASS: $elf"
