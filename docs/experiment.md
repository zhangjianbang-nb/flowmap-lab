# 实验协议：算法工程实验室

本文件把“跑一个 demo”升级为可复现的实验设计：假设、变量、指标、步骤、消融、失败判据。所有实验都可在单机 CPU/MPS 上完成。

## 1. 研究问题

1. FLM 的 flow-matching 目标能否把噪声搬到多峰数据流形？
2. FMLM 的一步映射能否在保持样本质量的同时，把采样成本从 N 次前向降到 1 次？
3. 概率路径（log-linear vs linear）如何影响 2-D 玩具与离散 token 任务？
4. 合成 Markov 语言上的 one-hot flow matching 是否能学到合法转移结构？

## 2. 假设

| 编号 | 假设 | 判据 |
|---|---|---|
| H1 | FLM 能覆盖 8 高斯环的所有 mode | `mode_coverage = 1.0` 且 `mode_hit_0.3 > 0.5` |
| H2 | FMLM 一步样本接近教师 8 步轨迹 | 平均余弦相似度 > 0.6；前向次数 1 vs 8 |
| H3 | linear 路径在 2-D 玩具上优于 log-linear | 相同步数下 `mode_mean_distance` 更低 |
| H4 | Markov one-hot FLM 学到合法 bigram | `valid_bigram_rate > 0.9`，token KL 明显低于均匀分布 |

> H3 是“控制变量”假设：论文的 log-linear 是为语言信噪比设计的，不代表它在低维几何任务上最优。消融结果应如实呈现，不得为了让论文路径好看而调参。

## 3. 变量

- 自变量：算法（`flm` / `fmlm`）、调度（`linear` / `loglinear`，`k∈{2,4,8}`）、训练步数、采样步数、模型容量。
- 因变量：`mode_hit_0.3`、`mode_mean_distance`、`mode_coverage`、`valid_bigram_rate`、`token_kl`、`elapsed_s`。
- 控制变量：seed、batch size、学习率、EMA 衰减、评估样本数。

## 4. 步骤

1. **数据检查**：确认 8 个 mode 中心、噪声尺度、Markov 合法转移集合。
2. **基线**：`configs/flm_tiny.yaml`，2000 步，记录 loss 曲线与 mode 指标。
3. **教师-学生**：`configs/fmlm_tiny.yaml`，先训教师 800 步，再蒸馏 1500 步；记录一步 vs 八步余弦。
4. **语言任务**：`configs/markov_flm.yaml`，1500 步；记录合法 bigram 率与 token KL。
5. **消融**：`configs/ablate.yaml`，算法 × 调度 k × 300 步；输出 CSV/Markdown。
6. **服务验证**：起 `serve.py`，`/health` 与 `/generate` 各请求一次，确认返回结构。

## 5. 消融矩阵

| 维度 | 取值 | 目的 |
|---|---|---|
| algo | flm, fmlm | 多步 vs 一步 |
| schedule.k | 2, 4, 8 | 信噪比斜率 |
| train_steps | 300, 1000, 3000 | 收敛性 |
| sample_steps | 1, 4, 8, 16 | 采样成本-质量曲线 |

建议先跑 6 组（2 算法 × 3 个 k），确认趋势后再扩展。每组固定 seed=0；最终报告用 seed∈{0,1,2} 的均值。

## 6. 失败判据

出现以下任一情况，实验结论标记为“不成立”，并在报告中写明：

- loss 不下降或发散（NaN/Inf）；
- `mode_coverage < 0.75`（漏 mode）；
- FMLM 一步余弦 < 0.4（学生没学到教师轨迹）；
- Markov 合法 bigram 率接近随机基线（约 2/16 = 0.125）。

## 7. 计算预算

| 任务 | 设备 | 预估 |
|---|---|---|
| 2-D FLM 2000 步 | CPU | < 5 s |
| 2-D FMLM 教师 800 + 蒸馏 1500 | CPU | 1–3 min |
| Markov FLM 1500 步 | CPU | 1–2 min |
| 消融 6 组 × 300 步 | CPU | < 1 min |
| 全量单测 | CPU | < 1 min（不含 FMLM 长测） |

## 8. 记录规范

- 每次运行写入 `runs/<name>/summary.json`、`train_log.csv`、`ckpt.pt`。
- 报告必须同时给出 raw 与 EMA 指标；EMA 带 warmup，短跑不得把初始化当收敛。
- 所有图表/表格从 `summary.json` 生成，不手抄数字。

## 9. 实测结论（seed=0）

| 假设 | 结论 | 证据 |
|---|---|---|
| H1 | 支持 | FLM linear 2000 步：`mode_hit_0.3=0.762`，`mode_coverage=1.0` |
| H2 | 支持 | FMLM 一步 vs 教师八步余弦 `0.998`；前向 1 次 vs 8 次 |
| H3 | 支持 | 同预算下 linear 的 `mode_mean_distance` 明显低于 log-linear（见 README 消融表） |
| H4 | **不支持** | Markov 1500/6000 步 `valid_bigram_rate≈0.12`，等于随机基线；只学到均匀边缘分布 |

H4 的失败不是实现错误，而是方法在小模型/短预算下的真实局限：one-hot 目标与高斯噪声的 flow matching 在离散转移结构上收敛很慢。后续可尝试：更大的模型、token embedding、CE 辅助损失、或论文规模的语料与训练预算。报告不得把 H4 写成通过。
