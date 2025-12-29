# PACD

## 测试与基线

### PACD-IVAPCI 流水线

运行端到端流程：

```bash
python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/pacd
```

常用参数：

- `--alpha`：CI 显著性水平
- `--max-k`：最大条件集大小
- `--estimator`：`ivapci`、`pacd` 或 `simple`
- `--epochs`：IVAPCI/PACD-T 训练轮数
- `--baseline-conds`：干预证据的基线条件（逗号分隔）
- `--effect-threshold`：固定剪枝阈值（默认使用分位数）
- `--effect-quantile`：分位数剪枝阈值
- `--device`：计算设备（`auto`/`cpu`/`cuda`）

#### 激活多环境信号（COND）

当数据包含 `COND` 列时：

- 如果未提供干预映射文件，脚本会自动将 `COND` 中非基线条件视为干预环境；
- `--baseline-conds` 用于指定哪些条件属于基线（默认 `CD3CD28,CD3CD28+ICAM2`）。

示例：

```bash
python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/pacd --baseline-conds CD3CD28,CD3CD28+ICAM2
```

#### 启用 GPU

默认 `--device auto` 会在有 GPU 时使用 `cuda`，否则回退到 `cpu`。

```bash
python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/pacd --device cuda
```

### PC 基线

如果已安装 `causal-learn`，运行 PC 基线：

```bash
python run_pc_baseline.py --data sachs_data.csv --output results/pc --alpha 0.001 --max-k 3
```

PC 基线输出：

- `skeleton.csv`：无向骨架边
- `cpdag.csv`：CPDAG 中的有向/无向边
- `pc_graph.json`：用于对比的 JSON 结果

未安装 `causal-learn` 时会提示安装并退出。

### 合成基准（PACD vs PC）

生成合成场景并对比骨架：

```bash
python run_synthetic_benchmark.py --output results/synthetic --n 1000 --n-vars 12 --alpha 0.001 --max-k 3
```

输出内容：
- 每个场景的 CSV 数据
- 真实边（`*_truth.json`）
- 汇总指标（`summary.json`）

### 结果报告

生成 PACD/PC/合成基准的 Markdown 报告：

```bash
python run_results_report.py --pacd results/pacd --pc results/pc --synthetic results/synthetic --output results/report.md
```
