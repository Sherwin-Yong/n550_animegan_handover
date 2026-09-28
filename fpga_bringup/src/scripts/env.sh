#!/usr/bin/env bash
# 在 WSL 中 source 本文件。
export ANIMEGAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ANIMEGAN_SDK="${ANIMEGAN_SDK:-/opt/animeganv3}"
export ANIMEGAN_BUILD="$ANIMEGAN_SDK/build"
export CC="$ANIMEGAN_SDK/tools/riscv-elf-toolchain/bin/riscv-unknown-elf-gcc"
export QEMU_RISCV64="$ANIMEGAN_SDK/tools/qemu_linux/bin/qemu-riscv64"
export ANIMEGAN_PYTHON="$ANIMEGAN_SDK/venv/bin/python"
export ANIMEGAN_MARCH=rv64gcv_zfh_zfbfmin_zvfh_zvfbfmin_zvfbfwma_xewmatrix1p0_zicbom_zicbop_zicboz_xdcache
export QEMU_CPU_FULL=rv64,matrix=true,v=true,vlen=1024,zfh=true,zfbfmin=true,zvfh=true,zvfbfmin=true,zvfbfwma=true
export PATH="$ANIMEGAN_SDK/venv/bin:$ANIMEGAN_SDK/tools/riscv-elf-toolchain/bin:$ANIMEGAN_SDK/tools/qemu_linux/bin:$PATH"
