# 源码与重新编译

本目录是源仓库提交 `0551b13b0cbd7ab93793d5b3a5f00687b9ab9501` 的快照（板上程序所需的全部源码）加模型包 `build/animeganv3_bf16.agp`，
能编出与 `../release/` 逐字节相同的镜像（打包时已在临时目录重编核对）。

## 目录

```
src/bsp/board.h          板级常量：地址、CLP 视图、mailbox 偏移、UART（时钟、波特率）
src/bsp/start.S          启动：BSS、ITCM 拷入、DTCM 清零、trap 入口
src/bsp/trap.c           异常服务（ITCM）：按 mcause 分派、打印现场
src/bsp/board.ld         链接脚本（DDR 镜像、.itcm、.dtcm）
tests/board_backdoor.c   板上主程序：magic、开机自检、前向、mailbox、逐节点表与附加阶段
tests/board_bench.c      附加阶段的微基准 B2–B5
src/conv_ame.c           AME 卷积；src/ops_rvv.c RVV 算子；include/animegan/ame.h 矩阵指令包装
include/animegan/board_log.h  阶段号与 BOARD_LOG 调试日志宏
scripts/build-board.sh   构建脚本；scripts/env.sh 工具链路径与 ISA 串
docs/n550_mem_map.md     CPU 视角地址映射
```

## 编译

工具链：`riscv-unknown-elf-gcc` 14.1.1（ESWIN，与参考工程相同），ISA 串见 `scripts/env.sh`，不要删减。
`scripts/env.sh` 取 `$ANIMEGAN_SDK/tools/riscv-elf-toolchain/bin/riscv-unknown-elf-gcc`（默认 `ANIMEGAN_SDK=/opt/animeganv3`），
输出到 `$ANIMEGAN_SDK/build/board/`。

```bash
source scripts/env.sh
BOARD=s2c bash scripts/build-board.sh           # 发布版 board_s2c_ame.bin（交付的镜像）
BOARD=s2c DEBUG=1 bash scripts/build-board.sh   # 调试版 board_s2c_ame_debug.bin（逐阶段、逐节点打印，不在交付包内）
```

- 另需主机 `cpp`、`awk`（构建脚本读 `board.h` 生成地址表）。
- `BOARD_EXTRA_DEFS='-D...'` 追加宏，例如 `-DBOARD_UART_CLK_HZ=<Hz>`、`-DBOARD_UART_BAUD=<baud>`。
- 改地址只改 `src/bsp/board.h`；张量数据须留在 CLP 段（只能 RVV 与 AMU 访问，标量访问会 trap）。
- 镜像须止于 mailbox `0x87F00000`，`trap_entry` 须在 ITCM 起址，构建脚本会检查。
