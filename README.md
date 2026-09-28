# n550 · AnimeGANv3 Ghibli-c1 bf16 上板交付

把 AnimeGANv3 Ghibli-c1 拍照风格化（512×512，bf16，AME 卷积 + RVV）跑在 S2C N550+AMU FPGA 开发板上。
主机经 PCIe 后门写入程序与一帧 RGB UINT8，放开复位，轮询 mailbox 后读回结果帧。

## 给 FPGA 上板 → [`fpga_bringup/`](fpga_bringup/)

从 [`fpga_bringup/README.md`](fpga_bringup/README.md) 开始读：

- 传输后先 `sha256sum -c checksums.sha256`。
- 先跑 `debug/`（串口逐阶段、逐节点打印），通过后跑 `release/` 取计时。
- 回传 mailbox、结果帧、输出张量与串口日志，由开发侧打分。
- 源码与重新编译见 `fpga_bringup/src/BUILD.md`；`BUILD_INFO.txt` 记录镜像对应的源仓库提交。

## 构建环境

```
march   = rv64gcv_zfh_zfbfmin_zvfh_zvfbfmin_zvfbfwma_xewmatrix1p0_zicbom_zicbop_zicboz_xdcache
mabi    = lp64d
cmodel  = medany
编译器  = riscv-unknown-elf-gcc 14.1.1（ESWIN）
```

本仓库内容由源仓库 `n550_AnimeGAN_bf16` 的 `scripts/pack-board.sh` 生成，不在此手改。
