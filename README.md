# flowmap-lab

[![CI](https://github.com/zhangjianbang-nb/flowmap-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/zhangjianbang-nb/flowmap-lab/actions/workflows/ci.yml)

一个**从零实现、可运行、可复现**的 FLM / FMLM 参考工程：连续 token 上的 flow matching、flow-map 蒸馏、一步生成、消融实验、HTTP 推理服务，全部包含在一个小型 Python 包里。

> **论文**：Flow Map Language Models: One-step Language Modeling via Continuous Denoising
> **URL**：<https://arxiv.org/abs/2602.16813>（PDF：<https://arxiv.org/pdf/2602.16813>）
>
> 这不是论文官方代码的复制品。官方实现（[david3684/flm](https://github.com/david3684/flm)）面向大规模语料与多机训练；本项目用**干净房间式重写**把同一套数学压缩到单机可跑的最小闭环，便于读懂、验证和二次实验。

## 它包含什么

| 模块 | 内容 |
|---|---|
| `flowmap/schedule.py` | log-linear / linear 概率路径，解析式与单调三次 LUT |
| `flowmap/model.py` | 2-D MLP 与 TinyTransformer，单/双时间条件，EMA（带 warmup） |
| `flowmap/flm.py` | 条件 flow-matching 损失、Euler 积分、多步采样 |
| `flowmap/fmlm.py` | 两时间 flow-map 蒸馏、一步采样、与教师轨迹的一致性评估 |
| `flowmap/data.py` | 8 高斯环（2-D 基准）与合成 Markov 语言（one-hot token） |
| `flowmap/train.py` | 训练循环、教师训练/加载、checkpoint、CSV 日志、评估 |
| `flowmap/eval.py` | 单 checkpoint 复评 |
| `flowmap/ablate.py` | 调度 × 算法 × 步数的消融矩阵 |
| `flowmap/serve.py` | 零框架 HTTP 推理服务（`/health`、`/generate`） |
| `tests/` | 调度、模型、数据、收敛、蒸馏、CLI、HTTP 共 24 个测试 |

## 快速开始

```bash
# 依赖：torch >= 2.0（CPU 即可），PyYAML 可选
python -m flowmap.train --config configs/flm_tiny.yaml --out runs/flm
python -m flowmap.train --config configs/fmlm_tiny.yaml --out runs/fmlm
python -m flowmap.train --config configs/markov_flm.yaml --out runs/markov

# 复评 / 采样服务
python -m flowmap.eval --ckpt runs/flm/ckpt.pt --ema
python -m flowmap.serve --ckpt runs/flm/ckpt.pt --port 8140
curl -s localhost:8140/health
curl -s -X POST localhost:8140/generate -H 'Content-Type: application/json' \
     -d '{"n":4,"steps":8}'

# 消融矩阵
python -m flowmap.ablate --config configs/ablate.yaml --out runs/ablation

# 测试
./scripts/run_tests.sh
```

## 数学约定

统一约定 **t=0 是纯噪声，t=1 是干净数据**：

```
x_t = alpha(t) * x_1 + sigma(t) * eps,     eps ~ N(0, I)
v_t = alpha'(t) * x_1 + sigma'(t) * eps
```

两种路径：

- **log-linear（论文语言路径）**：`alpha(t)=exp(-k(1-t))`，`sigma(t)=1-alpha(t)`，
  `gamma(t)=log(alpha/sigma)=-k(1-t)`，逆变换解析可得。
- **linear（rectified flow 基线）**：`alpha(t)=t`，`sigma(t)=1-t`。

FLM 的训练目标就是回归条件速度场：

```
L_FLM = E_{t~U(0,1), eps~N(0,I)} || v_theta(x_t, t) - (alpha'(t) x_1 + sigma'(t) eps) ||^2
```

FMLM 学习两时间映射 `f_theta(x_t, t, s)`（`s > t`），教师是训练好的 FLM：

```
x_s^teacher = Euler(v_FLM, x_t, t -> s)
L_FMLM = E || f_theta(x_t, t, s) - sg(x_s^teacher) ||^2
```

推理时一次前向：`x_1 = f_theta(x_0, 0, 1)`。推导细节见 [`docs/derivation.md`](docs/derivation.md)。

## 实测结果

在 Mac mini M6（24 GB）上实跑，`runs/*/summary.json` 会记录每次运行的完整指标。下表是 seed=0 的单次结果，括号内为 EMA 权重。

| 实验 | 配置 | 步数 | 关键指标 |
|---|---|---|---|
| FLM · 8 高斯环 | `configs/flm_tiny.yaml`（linear） | 2000 | `mode_hit_0.3=0.762`，`mode_coverage=1.0`，`mode_mean_distance=0.212` |
| FMLM · 8 高斯环 | `configs/fmlm_tiny.yaml`（linear，教师 1500 + 蒸馏 1500） | 1500 | `mode_hit_0.3=0.648`，`mode_coverage=1.0`，一步 vs 教师八步余弦 **0.998** |
| FLM · Markov 语言 | `configs/markov_flm.yaml`（causal，log-linear k=4） | 1500 / 6000 | `valid_bigram_rate=0.12`（随机基线 0.125），**负结果**：只学到均匀边缘分布 |
| 消融 · log-linear k | `configs/ablate.yaml`（1000 步） | 1000 | 见下表 |

消融（EMA 指标，`runs/ablation/ablation.md` 自动生成）：

| algo | k | mode_hit_0.3 | mode_mean_distance | coverage_0.3 |
|---|---|---|---|---|
| flm | 2.0 | 0.330 | 0.440 | 0.783 |
| flm | 4.0 | 0.174 | 0.590 | 0.647 |
| flm | 8.0 | 0.045 | 0.879 | 0.277 |
| fmlm | 2.0 | 0.129 | 0.639 | 0.541 |
| fmlm | 4.0 | 0.047 | 0.916 | 0.283 |
| fmlm | 8.0 | 0.049 | 0.940 | 0.260 |

**读法**：在低维几何任务上，log-linear 的 k 越大（t=0 处噪声越纯），条件速度场越难学；linear 路径（rectified flow）在同样预算下明显更好。Markov 任务在 626k 参数 / 6000 步下仍只学到边缘分布，说明 one-hot + 高斯噪声的 flow matching 在小模型上不足以恢复离散转移结构——这是本项目的负结果，不作为“已解决”宣称。

复现实验：

```bash
python -m flowmap.train --config configs/flm_tiny.yaml --out runs/flm-2k --steps 2000
python -m flowmap.train --config configs/fmlm_tiny.yaml --out runs/fmlm-1.5k --steps 1500
python -m flowmap.train --config configs/markov_flm.yaml --out runs/markov-1.5k --steps 1500
python -m flowmap.ablate --config configs/ablate.yaml --out runs/ablation
```

## 目录结构

```
flowmap-lab/
├── flowmap/            # 包：调度/模型/损失/训练/评估/服务/消融
├── configs/            # YAML 配置
├── tests/              # unittest，无 pytest 依赖
├── docs/               # 推导与实验协议
├── scripts/            # 测试与演示脚本
└── .github/workflows/  # CPU CI
```

## 设计取舍

1. **单机可跑**：默认 2-D 8 高斯环，CPU 数秒完成一次训练；语言任务用 16 词表 Markov 链，避免下载语料。
2. **指标可判别**：8 高斯环用 mode hit rate / mode coverage，Markov 用合法 bigram 率与 token KL，而不是难以解释的代理困惑度。
3. **诚实标注**：EMA 带 warmup；短训练不会把随机初始化当成“收敛”。log-linear 路径在 2-D 玩具上明显难于 linear，这一点在消融中如实呈现。
4. **零框架服务**：`serve.py` 只用标准库，便于在 CI 与容器中起服务。

## 限制

- 不是大规模语言模型训练代码：没有分布式、FSDP、flash-attn、数据管道。
- 2-D 与 Markov 任务只验证算法闭环，不代表真实文本生成质量。
- flow-map 蒸馏的上限由教师决定；教师未充分收敛时，一步图会继承其模糊性。

## 发布校验

首次发布后做了三遍独立校验，结果如下（2026-11-26）：

| 轮次 | 校验方式 | 结果 |
|---|---|---|
| 1 | GitHub API 读取 `main` 树，与本地 `git ls-tree -r HEAD` 逐文件比对 blob SHA | 34/34 完全一致 |
| 2 | 从 GitHub 重新克隆到干净目录，跑 `python -m unittest discover -s tests` | 24 项全部 OK（16.9 s） |
| 3 | 查 Actions 运行记录 + 远端 README/CI 配置可读性 | CI `completed/success`，提交 `818da6f` |

更新后 CI 徽章应显示为 passing；若失败，以 Actions 页面为准。

## 许可

MIT。见 [`LICENSE`](LICENSE)。
