# hrv-packet-loss-benchmark
Block-loss → RR-interpolation → HRV frequency-error benchmark on EEGMAT (72 public ECG segments). Compares three interpolation heuristics against a no-recovery baseline over 4 block lengths × 5 loss rates × 3 seeds. Reports per-method safe loss-rate thresholds for LF/HF/RMSSD and stress-decision flip rates.



# ECG 插值分析对比（丢包 → 补间期 → HRV 频域误差）

完整报告见 [out/README.md](out/README.md)：数据出处与许可、复现步骤、管线口径、结果表与三句话结论。

快速开始：

```bash
cp eegmat/*.npy data/eegmat/     # 副本已校验（72 文件 SHA-256 与源一致）
python src/run_experiment.py     # 全网格 ~50 s
python src/summarize.py          # 汇总 + 图
```

目录：`data/eegmat/` 只读副本；`src/` 检测器/掩码/插值/HRV/主脚本/汇总；`out/` 结果与报告。
