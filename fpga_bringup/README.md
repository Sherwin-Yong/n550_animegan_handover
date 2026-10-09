# 上板交付单：S2C N550+AMU bf16 后门加载镜像

面向执行上板的 FPGA 人员与分析回传结果的开发者。板端程序是裸机 raw bin，模型包已内嵌；CPU 处于复位期间，主机经 PCIe 后门把程序、自证 magic 与一帧 512² RGB UINT8 写入板上 DDR，然后放开复位；板上完成开机自检、归一化、Ghibli-c1 bf16 推理与反归一化，把结果帧写回 DDR、把状态与分阶段周期数写进 mailbox，主机轮询 mailbox 后读回结果帧与输出张量。完成状态写出后，板上继续在串口打印逐节点周期表并执行附加阶段（第二遍前向与比较、微基准），以 `extra done` 结束。每帧启动一次。文字信息全部走串口。

## 1 交付物

本说明是交付包 `fpga_bringup/` 的 README（由源仓库 `scripts/pack-board.sh` 生成，源文件 `docs/board-delivery.md`）；下文路径相对 `fpga_bringup/`。

| 文件 | 内容 |
| --- | --- |
| `release/board_s2c_ame.bin` | 唯一的 raw 镜像，写到 `0x80000000`；计时取自它，串口内容见 §4 |
| `release/` 下的 `.elf`、`.diss`、`.sym` | 带符号镜像、反汇编（含源码行）、按地址排序的符号表：按串口的 `mepc`/`ra` 查出错函数与源码行 |
| `input/test8.input.bin` | 固定输入帧（竖幅照片 `test8.jpg` 整帧拉伸到 512²），786432 B，写到 `0xF0000000` |
| `addresses.txt` | 加载地址表 |
| `BUILD_INFO.txt` | 源仓库提交、编译器、ISA 串与构建命令 |
| `src/` | 源码快照与模型包，重新编译见 `src/BUILD.md`，能编出与 `release/` 逐字节相同的镜像 |
| `MANIFEST.txt`、`checksums.sha256` | 文件清单与校验和：传输后先执行 `sha256sum -c checksums.sha256`，全部 OK 才继续 |

- 只跑这一个镜像、只跑一次：主结果、逐节点周期与附加数据都在这一次的 mailbox、读回文件与串口日志里。
- 模型包已内嵌进镜像，无需另外加载权重。

## 2 地址与格式

| 区域 | 地址（主机视图） | 主机动作 | 内容 |
| --- | --- | --- | --- |
| 程序 | `0x80000000` | 写 | raw bin；镜像（含 BSS 与栈）止于 mailbox 之前，构建脚本检查 |
| mailbox | `0x87F00000`，4 KB | 写状态字、读 | 见下表 |
| PCIe 自证 magic | `0x87F00080` | 写 | 32 位立即数 `0x5A5AC3C3`（不是文件）；板上读回并回显到 `0x87F00084` |
| 输入帧 | `0xF0000000` | 写 | 786432 B：512×512 RGB UINT8，行优先 HWC |
| 输出帧 | `0xF0100000` | 读 | 786432 B，同格式 |
| 输出张量 | `0xF0400000` | 读 | 1572864 B：bf16 图输出张量，稠密 NHWC；开跑前清零；mailbox `+0x38`/`+0x40` 同报地址与字节数 |

mailbox（全部小端）：

| 偏移 | 字段 | 说明 |
| --- | --- | --- |
| `+0x00` | u32 状态 | `0xB0000001` 启动、`0xB0000003` 运行、`0xB0000004` 完成、`0xB00000FF` trap（这四个与参考工程相同）；`0xB00000EE` 出错为本程序新增；参考工程的 `0xB0000002` 就绪本程序不用 |
| `+0x04` | u32 错误码 | 1 magic 不符、2 模型包打开失败、3 内存分配或 CLP 布局不符、4 图输入输出不符、5 内存规划失败、6 前向失败、8 开机自检失败 |
| `+0x08` | u32 cpu_hz | 40000000 |
| `+0x0C` | u32 节点数 | 79 |
| `+0x10` | u64 × 5 | 周期数：pre、forward、conv、other、post；秒数 = 周期数 / cpu_hz |
| `+0x38` | u64 × 2 | 输出张量 DDR 地址、字节数（固定为 `0xF0400000`、1572864，与上表相同） |
| `+0x48` | u64 × 6 | trap 时的 mcause、mepc、mtval、mstatus、ra、sp |
| `+0x78` | u32 × 2 | 阶段号、节点号（阶段：1 启动、2 开机自检、3 模型与规划、4 清零、5 pre、6 forward、7 post、8 完成；附加阶段 9 第二遍前向与比较、10 微基准、11 trap 自检，只在 trap 时写入 mailbox） |
| `+0x80` | u32 | 主机写入的 magic（板上读完后清零，每帧须重写） |
| `+0x84` | u32 | magic 回显 |

- 成败只看 `+0x00` 状态字；参考工程的 FAILMASK（`+0x1C`）与 EPOCH（`+0x14`）在本程序不适用，这两个偏移落在周期数字段内。

| 项 | 值 | 来源 |
| --- | --- | --- |
| CPU/AMU 频率 | 40 MHz | `src/bsp/board.h` |
| 串口 | DW_apb_uart 16550 兼容，基址 `0x20100000`，115200 8N1，输入时钟 10 MHz | 同上 |
| 缓存一致性 | 张量数据只由 RVV 经不缓存的 CLP 窗口与 AMU 访问；标量写 mailbox 后 `l1d_clean_all` | `src/platform_board.c` |

## 3 执行步骤

- S01 校验：`sha256sum -c checksums.sha256` 全部 OK。
- S02 先接好串口并开始保存串口输出（从放开复位前开始，到 S04 结束）；拉住 CPU 复位（或断电重上后保持复位），写入期间保持复位；PCIe 后门写入：程序 `release/board_s2c_ame.bin` → `0x80000000`；`0x00000000` → `0x87F00000`（清掉上一帧的状态字）；`0x5A5AC3C3` → `0x87F00080`；`input/test8.input.bin` → `0xF0000000`。
- S03 放开复位（硬件复位或断电重上：软复位后 L1D 残留的脏行可能在启动时写回，覆盖刚写入的数据）；轮询 `0x87F00000` 直到 `0xB0000004`（完成）、`0xB00000EE`（出错）或 `0xB00000FF`（trap）。
- S04 读回：`0x87F00000` 起 4096 B 存为 `mailbox.bin`；`0xF0100000` 起 786432 B 存为 `board_out.bin`；`0xF0400000` 起 1572864 B 存为 `board_out.tensor.bin`（即 mailbox `+0x38`/`+0x40` 给出的地址与字节数）。读回后继续保存串口，直到出现 `extra done`，或完成后 60 s 内没有新输出；然后再读一次 `0x87F00000` 起 4096 B 存为 `mailbox_end.bin`（附加阶段出 trap 时这里是 `0xB00000FF` 与现场）。
- S05 回传 §5 所列文件，由开发侧打分（§7）。
- S06 下一帧：从 S02 起重做一遍，程序也要重写（程序的 `.data` 带初值，只在写入程序时恢复）。
- S07 拍照整链（与开发侧配合）：开发侧客户端写出 `input.bin`，操作者按 S02–S04 用它替换输入帧写入并读回，把结果帧交回为 `output.bin`。

## 4 判据与预期

- 板上完成：mailbox 状态 `0xB0000004`、magic 回显 `0x5a5ac3c3`；串口有 `clp self-test ok` 与 `done`。
- 通过（开发侧，§7）：结果帧对参考图 `PSNR ≥ 45 dB` 且 `max|d| ≤ 6`，输出张量 `max|d|` ≤ 5% × `max|ref|`，末行 `BOARD CHECK PASS`。
- 预期耗时：前向以 mailbox 周期数为准（主前向的计时区间内没有串口打印）；2026-09-30 实测前向 8.13 s（卷积 6.11 s、其它 2.01 s），本版目标 ≤ 3.00 s。
- 串口内容，按出现顺序：
  - 启动横幅、`pcie magic … ok`、`clp self-test ok`；
  - 主前向期间的进度标记 `#<阶段>;`、`@<节点>;`：只在串口空闲时写、从不等待，可能缺失，卡住时以最后一个完整标记定位；
  - `cycles pre=… forward=… conv=… other=… post=… cpu_hz=…`、`done`（此时 mailbox 已是完成状态）；
  - `per-node cycles, main forward:` 下 79 行 `node <i> op <opcode> cycles <c>`；
  - 附加阶段：第二遍前向（关闭 P05 重叠）期间每个节点一行 `stats node <i> sum_u … abs_u … max_u …`（共 79 行）、`extra cycles conv=… other=… (overlap off; node sums: the pass total … includes the statistics output)`、`per-node cycles, extra forward (overlap off):` 下 79 行、`extra: second forward result frame vs main result frame: identical`（或不同的字节数与首个位置）、`bench B2 …` 至 `bench B5 …`、`trap self-test ok: breakpoint at mepc=…, resuming after it`，最后 `extra done`。
- 附加阶段在完成状态之后运行，不改动结果帧、输出张量与 mailbox 的周期字段；它出错或 trap 时主结果照常有效，只是附加数据缺失。

## 5 回传物

- `mailbox.bin`（4096 B）、`board_out.bin`（786432 B）与 `board_out.tensor.bin`（1572864 B；多读的部分打分时忽略）原始字节；`mailbox_end.bin`（S04 末尾再读的 4096 B）。
- 串口原始日志，从横幅到 `extra done`（或停止输出处）完整保存，不删改。
- 实际写入地址、写入文件的哈希与任何重建参数。

## 6 故障对照

| 现象 | 处理 |
| --- | --- |
| 状态一直是 0 或写入前的值，串口无横幅 | CPU 没跑起来：查复位是否放开、程序是否写到 `0x80000000` |
| 串口乱码 | 查波特率 115200 8N1 与 UART 输入时钟（10 MHz；不同时以 `BOARD_EXTRA_DEFS=-DBOARD_UART_CLK_HZ=<Hz>` 重建） |
| `0xB00000EE`、错误码 1、回显不是 `0x5a5ac3c3` | PCIe 写入没进 DDR 或写在复位放开之后：查 magic 地址与写入顺序 |
| `0xB00000EE`、错误码 8 | 开机自检失败（RVV 经 CLP 窗口写入的数据与 AMU 读写不一致）：回传串口中 `clp self-test: C[…]` 行与 mailbox，停止后续帧 |
| 开机自检阶段（阶段号 2）trap，非法指令且 `mepc` 落在 `mlae16`/`mlbe16`/`msce32` 上 | 矩阵指令的寄存器形式不被接受（A tile 用 a2/a3 装载已于 2026-09-30 调试版验证可用）：回传串口 trap 整段，用 `.diss` 核对该指令的寄存器，停止后续帧 |
| `0xB00000EE`、其它错误码 | 回传串口输出（`error <码>: <原因>` 行）与 mailbox |
| `0xB00000FF` | trap：回传串口中 `*** TRAP: …` 起的整段与 mailbox（`+0x48` 起）。首行是要点（异常名、mcause、mepc、mtval、阶段、节点号与 opcode），写 mailbox 之前已发出；之后是完整现场（提示、地址区域与 CLP 槽位偏移、CSR 与 mcycle、vl/vtype/vstart、mtilem/n/k、mepc 处指令字、全部寄存器、sp 起 32 个双字）。非法指令会先以 `fence.i` 重试至多 3 次，串口有对应行。`*** NESTED TRAP` 行同时给出本次与第一次 trap 的 mcause/mepc/mtval |
| 附加阶段停在阶段 11，串口是 `*** TRAP: breakpoint …` 而不是 `trap self-test ok` | trap 服务能报告但自检没被识别；主结果有效，回传整段串口 |
| 先 `0xB0000004`，之后 `mailbox_end.bin` 为 `0xB00000FF`，或串口在 `done` 之后停止、没有 `extra done` | 附加阶段出错：主结果照常读回与打分；回传完整串口日志与两份 mailbox，阶段号 9–11 指出停在附加阶段的哪一步 |
| 状态停在 `0xB0000003` | 程序在前向中停住：回传 mailbox（`+0x78` 阶段号、节点号）与串口输出（最后一个 `@<节点>;` 标记） |

## 7 开发侧（源仓库）

- 出包：提交后在 WSL 执行 `source scripts/env.sh && bash scripts/pack-board.sh`，写入发布仓库 `n550_animegan_handover/fpga_bringup/`（`HANDOVER_DIR` 可改），镜像由脚本从已提交的源码构建，源码快照重编须与之逐字节相同。
- 打分（仓库根的客户端环境，用回传的三个文件；参考取 test8 的参考目录）：

```bash
apps/ubuntu-webcam/.venv/bin/python apps/ubuntu-webcam/board_check.py --output board_out.bin --mailbox mailbox.bin --tensor board_out.tensor.bin --reference-dir build/reference-test8
```

- 打分 `BOARD CHECK FAIL` 时：以回传的结果帧、输出张量、mailbox 与串口日志（两遍前向的逐节点周期、第二遍与主结果的比较）定位；调试版（`DEBUG=1`）不在交付包内。
- 串口日志总览：`python tools/check.py boardlog <串口日志> [--baseline <旧日志>] [--reference-dir build/reference-test8] [--elf fpga_bringup/release/board_s2c_ame.elf]`（WSL，`source scripts/env.sh` 后；`--elf` 缺省为本机构建目录的 s2c 发布版镜像，分析回传日志时指定交付包里的那个）。

### 7.1 故障定位

| 现象 | 看哪几行 | 定位命令 | 修改切入点 |
| --- | --- | --- | --- |
| trap（`0xB00000FF`） | `*** TRAP:` 要点行：异常、mepc、mtval、阶段、节点号与 opcode；完整现场：地址区域与 `CLP offset`、`instruction at mepc`、`vector:`、`matrix:`、`registers:`、`stack from sp` | `check.py boardlog <日志> --elf <交付包 elf>`：mepc、ra 与栈里的程序地址解析成函数与源码行（含内联链）；`CLP offset … arena +…` 解析成该节点时存活的张量（`runtime_test <模型包> --plan`） | mepc 所在函数；opcode 3 → `src/conv_ame.c`（打包首层看 `AG_TENSOR_LANE_PACK` 分支），100/0/98/23/28/200 → `src/ops_rvv.c` 的 pad/elementwise/resize/tanh/lade；越界地址对照该张量的字节数与 `ag_view`；`vstart` 是出错元素序号 |
| 嵌套 trap | `*** NESTED TRAP` 行（本次与第一次 trap 的 mcause/mepc/mtval，阶段/节点） | 同上，用第一次的 mepc | 第一次 trap 的位置；本次 mepc 在 ITCM（`0x0400xxxx`）即 `src/bsp/trap.c` 报告代码 |
| 出错（`0xB00000EE`） | `error <码>: <原因> <值> (stage … node …)`；码 6 另有 `node … opcode … failed rc=…` | §2 错误码表 | `tests/board_backdoor.c` 中对应的 `fail()`；码 6 的 rc 来自该 opcode 的内核（`src/conv_ame.c` 的 -28/-29、`AG_NO_KERNEL` -30 即 RVV 内核不支持该形状） |
| 卡住（状态停在 `0xB0000003`，或 `done` 后无 `extra done`） | 最后一个完整的 `#<阶段>;`、`@<节点>;` 标记；mailbox `+0x78` | `check.py boardlog` 的 `last progress markers` | 该节点的内核；卷积卡住先看 `src/conv_ame.c` 的 fence 与 tile 参数（B4 的结果可对照） |
| 结果不符（`BOARD CHECK FAIL`） | 附加阶段的 `stats node …` 行；`extra: second forward result frame vs main result frame` | `check.py boardlog <日志> --reference-dir build/reference-test8`：偏离最大的节点与第一个超阈值的节点 | 第一个超阈值节点的内核；第二遍（关重叠）与主结果不同 → `src/conv_ame.c` 的重叠与 C 双缓冲（`ag_conv_overlap`） |
| 耗时异常 | 两张逐节点周期表；`bench B2`–`B5` | `check.py boardlog <日志> --baseline <旧日志>`：逐节点与按算子的变化、重叠开/关对比 | 变慢的节点的内核；重叠无效看 B4；尾处理看 B3；单条指令代价看 B2 |
| trap 自检没打印通过行 | 阶段 11 的 `*** TRAP: breakpoint` | — | `src/bsp/trap.c` 的自检分支、`src/bsp/start.S` 返回路径恢复 mepc |
- 出图（同一环境）：`apps/ubuntu-webcam/.venv/bin/python apps/ubuntu-webcam/board_image.py --output board_out.bin --image archive/test8.jpg`，写 `board_out.png`（512² 原样）、`test8.styled.png`（按拍照客户端后处理拉回原图尺寸）与 `test8.panel.png`（原图 | 效果图）；只读回张量时改用 `--tensor board_out.tensor.bin`。

- 输入帧 `build/photos/test8.input.bin`（sha256 `a94f510dc2ba8c7657fad659b037eef061f697270ea141ff702625e3e56603e4`）：本机资料 `archive/test8.jpg`（sha256 `eb3d679ab503bc354fc20e1346c1c83353d832f863a42d053983289e3147eea6`，不入库）经拍照客户端整帧拉伸，`apps/ubuntu-webcam/.venv/bin/python apps/ubuntu-webcam/anime_cam.py --image archive/test8.jpg --backend onnx --save-dir build/photos --no-window`（同时写出 onnx 参考输出 `test8.output.bin`）。
- 参考目录 `build/reference-test8`：`python scripts/make_reference.py --input build/photos/test8.input.bin --out build/reference-test8`（约 900 MB）；参考图 `output_u8.bin` sha256 `a29fe26fb73fa4a716b2a81e6247ab2051181caa956015df0d29b913104837f1`，与 `build/photos/test8.output.bin` 相同。仓库回归用的 `reference-data/`（v3_3）不变。
- 模型包 `build/animeganv3_bf16.agp`（sha256 `f3e101ef774d262e64bd8408e589d0eb4972a44efc32ffffce1e5d96dda22f15`，`python tools/model_pack.py`）；模型 `models/AnimeGANv3_large_Ghibli_c1_e299.onnx`（sha256 `ad950eabe2a556424b1cacf0f4b7847777ea44033482e673771da878054c1e37`）。
- 拍照整链：`bash apps/ubuntu-webcam/run.sh --camera 0 --photo --backend file --exchange-dir <目录> --timeout 1800`。
- QEMU virt 镜像 `BOARD=qemu_virt [DEBUG=1] bash scripts/build-board.sh`（无 CLP 窗口，两视图相同）只保证可编译：QEMU virt 在 `0x04000000/0x04020000` 没有 ITCM/DTCM，镜像启动拷入 `.itcm` 即 store fault。
