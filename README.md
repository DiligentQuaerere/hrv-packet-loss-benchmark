# 丢包对 HRV 频域估计的影响：插值补间期基准（ECGMAT）

**HRV under packet loss — an RR-interpolation benchmark on public ECG segments**

> 该目录代码逻辑：把干净的 ECG 成块删掉，再用几种做法把缺掉的 R 峰间期补回来，看 HRV 指标误差有多大、从哪个丢包率开始不能用。

数据全部来自**公开数据集**（PhysioNet EEGMAT v1.0.0 / Fantasia v1.0.0，ODC-By 1.0）。本仓库**不包含任何原始生理信号**，只放代码与汇总结果。

---

## 1. 主要结论（全部可在 `results/` 里查到）

1. **不补间期完全不可用。** 5% 成块丢包下，HF 功率的中位相对误差约 **1840%**（≈18 倍），50% 丢包时到 **9218%**。补间期不是优化，是必需步骤。
2. **但补了也只在"块不碎"的时候能用。** 名义丢包率 5%、块长 2 s 时 HF 中位误差 **62.5%**、LF **151.0%**；块长换成 5 s 立刻降到 6.4% / 15.3%，10 s 为 5.2% / 5.5%。
3. **决定误差的是块长（断点次数），不是丢包总量。** 同样丢 5%，碎成 2 秒一块能把 HF 误差放大 12 倍、LF 放大 27 倍。
4. **62 秒窗里 LF 本身不是稳定量。** 把干净记录从 182 s 截成 62 s（无丢包、无插值），LF 的中位变动就有 **43.5%**（84% 的窗超过 10%）。→ 短窗应报告 RMSSD / SDNN / HF，**不要报告 LF / LF-HF / TP**。
5. **压力判定用 HF 比用 LF/HF 稳。** 用 HF 做"压力升高"的二分，三种插值在 5–10% 丢包下翻转率 11–14%；用 LF/HF 则是 25–31%。
6. **方法取舍**：`random_sd`（在均值 ±1SD 内随机抽）在 10–30% 丢包下 HF / RMSSD 误差最低；`mean_global` 在高丢包下 LF/HF 最稳；`mean_local` 在 50% 丢包会灾难性失败（HF 中位 **+162%**）。

## 2. 数据来源与许可 / Data & license

| 数据集 | 用途 | 出处 | 许可 |
|---|---|---|---|
| **EEGMAT v1.0.0** | 主实验（72 段 ECG，36 人 × 2 段） | PhysioNet <https://physionet.org/content/eegmat/1.0.0/>（Zyma 等，2019） | ODC-By 1.0 |
| **Fantasia v1.0.0** | 检测器一致性验证 | PhysioNet <https://physionet.org/content/fantasia/1.0.0/>（Iyengar 等） | ODC-By 1.0 |

- EEGMAT：500 Hz、int16 原始数字计数，`_1` 为约 182 s 静息段、`_2` 为 62 s 心算段。峰检测直接用数字计数，只影响幅值、不影响峰时刻。
- **不重新分发数据**：请自行从 PhysioNet 下载，放到 `data/eegmat/`（72 个 `.npy`）与 `data/fantasia/`（WFDB `.ecg`/`.hea`）。
- **"参考标注"的准确说法**：Fantasia 附带的是 `corrected beat annotations from an automated detector`——自动检测器输出并经校正，**不是人工逐峰标注**。
- 署名要求（ODC-By）：引用时请注明数据集名称、作者、PhysioNet 链接与本仓库。

## 3. 方法

**破坏模型**：在时间轴上按"块长 × 丢包率"删除若干互不重叠的块，块内用**线性桥接**填充（两端样本连线，无阶跃边缘，检测器会漏掉块内全部搏动）。这模拟的是"接收端 concealment"，**不是**真实的时钟漂移/丢采样。

**检测器**：自实现 Pan–Tompkins（5–15 Hz 二阶 Butterworth 零相位 → 五点差分 → 平方 → 150 ms 移动窗积分 → 双阈值自适应 + searchback + T 波斜率判别；不应期 250 ms）。完整信号与破坏信号用**同一个**检测器；`peaks_full` 即真值。

**四种方法**（签名统一 `recover(peaks, gaps, fs, rng)`，插入峰桥接缺口两侧存活峰，个数 `floor(块长/RR估计)`）：

| 代号 | 做法 |
|---|---|
| `none` | 不补，缺口保留为一个长 RR（基线） |
| `mean_global` | RR 估计 = 全部完好间期均值，等间隔铺满缺口 |
| `mean_local` | RR 估计 = 缺口左右紧邻完好间期的均值（缺侧回退全局） |
| `random_sd` | RR 抽自 `U(均值−SD, 均值+SD)`，按比例缩放铺满缺口 |

**HRV 口径**：RR tachogram 三次样条重采样 4 Hz → **生理 clamp [0.3, 2.0] s** → 去均值 → Welch（Hann，`nperseg=min(256,N)`，50% 重叠）→ 频带边界插值梯形积分。LF 0.04–0.15 Hz、HF 0.15–0.40 Hz、TP=LF+HF。时域含 SDNN / RMSSD / pNN50。

**评价**：相对误差 `(估计−真值)/真值`；结论一律用**中位数**（均值对离群极敏感）；另有方向一致率、以及"压力判定翻转率"（用 `_2`/`_1` 配对，HF 升高或 LF/HF 升高视为压力升高）。阈值定义为"中位绝对相对误差首次超过 10% / 20% 的名义丢包率"。

## 4. 结果文件怎么读

| 文件 | 内容 |
|---|---|
| `results/summary_{excluded,all}_{pooled,seg1,seg2}.csv` | 主结果：方法 × 丢包率 × 指标，含 n / 偏差 / 中位绝对相对误差 / 方向一致率 / 翻转率 |
| `results/summary_by_block_*.csv` | 同上，再按块长拆开（**块长是最强的效应**，这张表最重要） |
| `results/thresholds_*.csv` | 各方法的 10% / 20% 安全丢包率 |
| `results/thresholds_compare_excluded_vs_all.csv` | QC 排除 vs 不排除的敏感性对比 |
| `results/window_control_*.csv` | 纯窗长对照（182 s 截 62 s，无丢包无插值） |
| `results/seg_contrast_*.csv` | `_1` 与 `_2` 在块长对齐后的对比 |
| `results/lf_attribution_*.csv`、`method_spread_*.csv` | LF 误差的来源分解与方法间跨度 |
| `results/lockout_audit.csv` | 72 条记录的检峰锁死体检（最长 RR 1.508 s，无一条 >2 s） |
| `results/fantasia_detector_*.csv` | 与参考标注的一致性：**20 条可验证记录**，池化匹配率 **0.9049**、精确率 0.9971、峰位中位偏差 8 ms；其中 2 条记录检峰锁死（匹配率仅 0.423 / 0.167），**剔除这两条后 18 条汇总匹配率 0.9896**、精确率 0.9973（另有 20 条 `f1*` 记录本地无波形，未验证） |
| `results/results_long_{excluded,all}.csv.gz` | 全量逐格结果（126,252 行 / 135,864 行） |
| `figures/` | 热图、阈值曲线、分层图、块长效应图 |
| `logs/` | 各脚本运行日志（路径已脱敏） |

## 5. 复现

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# 或: python -m venv .venv && .venv/bin/pip install -r requirements.txt # Linux/macOS

# 1) 数据：自行从 PhysioNet 下载，放成
#    data/eegmat/Subject00_1.npy ... Subject35_2.npy   (72 个)
#    data/fantasia/*.ecg *.hea                          (可选，仅验证用)

python src/run_experiment_v2.py     # 主网格（约 1 分钟）
python src/summarize_v2.py          # 汇总表与图
python src/window_control.py        # 纯窗长对照
python src/seg_contrast.py          # _1 与 _2 对比
python src/fantasia_validate.py     # 检测器一致性（需 Fantasia）
python src/lockout_audit.py         # 检峰锁死体检
```

**随机性完全可复现**：掩码种子 = `sha256("eegmat-loss-v1|{subject}|{block}|{rate}|{rep}|mask")` 前 8 字节 → `np.random.default_rng`；插值用同键加后缀 `|recover`。同受试者的两段共用同一种子（配对翻转率需要）。

## 6. 局限（请连同结论一起读）

1. **真值不是人工标注**：`peaks_full` 来自本项目自实现的 Pan–Tompkins，不是专家逐峰标注。与参考标注的一致性只能说明"与另一检测器接近"。
2. **检测器有已知失效模式**：阈值单向更新、无重初始化路径，个别记录（如 Fantasia `f2o05`/`f2y05`）会在运行中途**锁死**。本仓库只做体检测试，未修改检测逻辑。
3. **长块 × 低丢包率组合不可行**（整数块量化），被跳过的格子不再补齐；各格实际丢包率见 `results/loss_actual_table.csv`。
4. **`_2` 段只有 62 s**：其频域绝对值噪声大，结论依赖同管线的相对误差。
5. **破坏模型是"接收端填值"**，真实丢包（缺采样、时基漂移）未模拟。
6. **QC 排除 5 条脏记录**（Subject02_1 / 13_1 / 20_2 / 28_1 / 35_2），敏感性分析见 `thresholds_compare_excluded_vs_all.csv`：215 项阈值对比里 **202 项（94%）完全不动**，其余 13 项各差一个名义档位。
7. **没有统计检验**：全部结论基于中位数与阈值档位，未做配对 Wilcoxon 或置信区间。
8. **只覆盖一个数据集**（EEGMAT）与一种设备/人群，外部效度有限。

## 7. 引用与致谢

- Zyma I, et al. *EEGMAT* (PhysioNet, v1.0.0), ODC-By 1.0.
- Iyengar N, et al. *Fantasia Database* (PhysioNet, v1.0.0), ODC-By 1.0.
- 检测器结构参考 Pan J, Tompkins WJ (1985) 的经典流程；HRV 指标口径参考 Task Force 1996。

## 8. 许可

代码：MIT（见 `LICENSE`）。数据：不随仓库分发，遵循各数据集原始许可。
