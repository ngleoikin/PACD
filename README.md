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
- `--direction`：结构学习方向方法（`pacd` / `mpcd`）
- `--mpcd-m-grid`：MPCD 尺度集合（逗号分隔）
- `--mpcd-stability-tau`：MPCD 稳定性阈值
- `--estimator`：`ivapci`、`pacd` 或 `simple`
- `--epochs`：IVAPCI/PACD-T 训练轮数
- `--n-bootstrap`：IVAPCI bootstrap 次数
- `--baseline-conds`：干预证据的基线条件（逗号分隔）
- `--effect-threshold`：固定剪枝阈值（默认使用分位数）
- `--effect-quantile`：分位数剪枝阈值
- `--device`：计算设备（`auto`/`cpu`/`cuda`）

#### 激活多环境信号（COND）

当数据包含 `COND` 列时：

- 如果未提供干预映射文件，脚本会自动将 `COND` 中非基线条件视为干预环境；
- `--baseline-conds` 用于指定哪些条件属于基线（默认 `CD3CD28,CD3CD28+ICAM2`）。

**基线环境解释：**
`COND` 列中被视为“未干预/对照”的条件集合。它们用于：
1) 作为干预效应比较的参照分布；
2) 自动构建干预映射时，排除这些条件（仅将非基线条件视为干预）。

如果你的数据中没有 `CD3CD28` 或 `CD3CD28+ICAM2`，请显式传入你的基线条件，例如：

```bash
python run_pacd_ivapci_pipeline.py --data your_data.csv --output results/pacd --baseline-conds controlA,controlB
```

示例：

```bash
python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/pacd --baseline-conds CD3CD28,CD3CD28+ICAM2
```

MPCD 结构学习示例（多尺度）：

```bash
python run_pacd_ivapci_pipeline.py \
  --data multienv_soft_low.csv \
  --output results/mpcd \
  --direction mpcd \
  --mpcd-m-grid 2,3,4,5 \
  --mpcd-stability-tau 0.6 \
  --baseline-conds env_0
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

提示：`run_pc_baseline.py` 会自动跳过非数值列（如 `COND`），并将结果写为 PACD 兼容的 JSON 格式（`pc_graph.json`），方便与 PACD 结果一起可视化对比。

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

### MPCD（多尺度渐进因果发现）

MPCD 在多个尺度上运行 PACD 骨架学习，并用稳定性选择过滤边，再做全局一致定向。
适合需要“跨尺度稳定边 + 风险补完方向”的场景。

示例（使用默认尺度集合）：

```bash
python - <<'PY'
import pandas as pd
from pacd_structure_learning import MPCDConfig, MPCDStructureLearner, PACDStructureConfig

df = pd.read_csv("sachs_data.csv")
data = df.select_dtypes("number").values
var_names = list(df.select_dtypes("number").columns)

base_cfg = PACDStructureConfig(alpha=0.001, max_k=3, m=4)
mpcd_cfg = MPCDConfig(m_grid=[2, 3, 4, 5], stability_tau=0.6, base_config=base_cfg)

learner = MPCDStructureLearner(mpcd_cfg)
result = learner.learn(data, var_names)

print("稳定骨架边数:", len(result["skeleton"]))
print("定向边数:", result["n_edges"], "未定向:", result["n_undirected"])
PY
```

常用参数说明：

- `MPCDConfig.m_grid`：尺度集合（例如 `[2,3,4,5]`）
- `MPCDConfig.stability_tau`：稳定性阈值（边在多少比例尺度出现才保留）
- `MPCDConfig.base_config`：内部 PACD 的配置（如 `alpha`、`max_k`、`ci_method` 等）

输出字段（`result`）：

- `skeleton` / `skeleton_indices`：稳定骨架
- `directed_edges`：定向与未定向边（`orientation_method` 标记）
- `edge_persistence`：跨尺度边频率（便于筛选/可视化）
- `scales` / `stability_tau`：本次 MPCD 的尺度与阈值配置

### S3C-DO（筛选-清洗-定向）

S3C-DO 是 “筛选 → 清洗（局部 PC-stable）→ 定向（v-structure + Meek）” 的三段式框架。
适合高维场景下先缩小候选邻域，再做局部化 CI 检验。

示例：

```bash
python - <<'PY'
import pandas as pd
from s3cdo_structure_learning import S3CDOConfig, S3CDOStructureLearner

df = pd.read_csv("sachs_data.csv")
X = df.select_dtypes("number").values
var_names = list(df.select_dtypes("number").columns)

cfg = S3CDOConfig(top_m=8, max_k=3, alpha=0.001, ci_method="spearman")
learner = S3CDOStructureLearner(cfg)
result = learner.learn(X, var_names)

print("候选边数:", len(result["candidate_edges"]))
print("骨架边数:", len(result["skeleton"]))
PY
```

常用参数：

- `top_m`：筛选阶段每个节点保留的候选邻居数
- `max_k`：清洗阶段最大条件集大小
- `ci_method`：`spearman` / `pearson`
- `use_nonparanormal`：是否做 copula 变换（增强线性化）

### 结果报告

生成 PACD/PC/合成基准的 Markdown 报告：

```bash
python run_results_report.py --pacd results/pacd --pc results/pc --synthetic results/synthetic --output results/report.md
```

### 定向 + IVAPCI 效应管线

使用 PC / PACD / MPCD / S3C-DO 定向，然后用 IVAPCI 估计每条边效应：

```bash
python run_direction_ivapci_pipeline.py --data sachs_data.csv --output results/dir_ivapci --direction pacd
```

S3C-DO 示例：

```bash
python run_direction_ivapci_pipeline.py \
  --data multienv_soft_low.csv \
  --output results/dir_ivapci \
  --direction s3cdo \
  --s3cdo-top-m 8 \
  --s3cdo-ci-method spearman \
  --s3cdo-ci-perm-samples 200 \
  --n-bootstrap 2
```

仅运行 S3C-DO 结构学习（不跑 IVAPCI）的脚本示例：

```bash
python - <<'PY'
import pandas as pd
from s3cdo_structure_learning import S3CDOConfig, S3CDOStructureLearner

df = pd.read_csv("multienv_soft_low.csv")
X = df.select_dtypes("number").values
var_names = list(df.select_dtypes("number").columns)

cfg = S3CDOConfig(
    top_m=10,
    alpha=0.01,
    max_k=3,
    ci_method="spearman",
    ci_perm_samples=200,
    collider_rule="cpc",
)
learner = S3CDOStructureLearner(cfg)
result = learner.learn(X, var_names)

print("candidate_edges:", len(result["candidate_edges"]))
print("skeleton:", len(result["skeleton"]))
print("directed_edges:", len(result["directed_edges"]))
PY
```

常用参数（完整说明）：

- `--direction`：方向学习方法。
  - `pc`：传统 PC（仅基于条件独立）。
  - `pacd`：p-adic PACD，默认推荐；单环境也可用。
  - `mpcd`：多尺度 PACD，适合希望稳定性更强的场景。
  - `s3cdo`：筛选-清洗-定向（S3C-DO），适合高维稀疏结构。
- `--alpha`：CI 显著性水平；越小越保守（删边更少）。
- `--max-k`：条件集最大大小；越大越耗时，且需要更多样本支撑。
- `--epochs`：IVAPCI 训练轮数（方向估计时使用）。
- `--device`：IVAPCI 训练设备（`cpu` / `cuda`）。
- `--n-bootstrap`：IVAPCI bootstrap 次数（不确定性估计，>0 时更稳但更慢）。
- `--only-structure`：仅做结构学习，输出 `directed_edges.csv/json`，跳过 IVAPCI 估计。

S3C-DO 相关（仅在 `--direction s3cdo` 时生效）：
- `--s3cdo-top-m`：筛选阶段每个节点保留的候选邻居数；大一些可降低漏边风险，但会增加后续计算量。
- `--s3cdo-ci-method`：CI 检验方法（`spearman` / `pearson`）。
  - `spearman` 更稳健但需置换检验；`pearson` 更快但对异常值敏感。
- `--s3cdo-use-nonparanormal`：启用 nonparanormal 变换；当变量分布明显非高斯时建议开启。
- `--s3cdo-ci-perm-samples`：Spearman 置换检验次数 `B`（默认 200）；越大 p 值分辨率越细但更慢。
- `--s3cdo-auto-fix-perm-resolution`：自动将 `alpha` 抬升到 `1/(B+1)` 的分辨率；
  - 可用 `--no-s3cdo-auto-fix-perm-resolution` 关闭（不推荐，可能导致过度保守）。
- `--s3cdo-collider-rule`：碰撞点规则（`naive` / `cpc` / `majority`）。
  - `cpc`：默认规则，优先避免错误定向；
  - `majority`：按多数分离集判断；
  - `naive`：只看第一组分离集（速度快但最不稳）。
- `--s3cdo-collider-majority-threshold`：`majority` 规则阈值（默认 0.5）。
- `--s3cdo-fallback-sepset-search`：对缺失 sepset 的三元组做补搜；
  - 当 `sepsets_all_` 缺失时，用邻域条件集补充独立证据。
- `--s3cdo-fallback-max-k`：补搜使用的最大条件集大小；为空则使用全局 `--max-k`。
- `--s3cdo-bootstrap`：S3C-DO 结构 bootstrap 次数（0 为关闭）；用于输出稳定性并做阈值筛边。
- `--s3cdo-bootstrap-threshold`：骨架稳定性阈值（默认 0.95）；低于阈值会丢弃该边。
- `--s3cdo-dir-threshold`：方向稳定性阈值（默认 0.95）；低于阈值的有向边会降级为无向边。

PACD/MPCD 相关（仅在 `--direction pacd/mpcd` 时生效）：
- `--mpcd-m-grid`：MPCD 的多尺度列表（如 `2,3,4,5`）。
- `--mpcd-stability-tau`：MPCD 稳定性阈值；越高越保守。

多环境/干预相关（当数据含 `COND` 列，或显式启用多环境逻辑时生效）：
- `--baseline-conds`：多环境基线条件（逗号分隔）；用于区分干预环境。
- `--intervention`：干预映射 JSON 文件（可选）；若不提供，会把非基线环境视为干预。

输出说明（重要）：
- `edge_effects.json` 里的 `p_value` 是 **效应显著性检验（ATE=0）** 的 p 值，不是 CI 条件独立检验的 p 值。
- S3C-DO 的条件独立证据会以 `ci_p_worst` 输出在每条边上（越小越稳健）。
- 若启用 `--s3cdo-bootstrap`，输出还会包含：
  - `skeleton_stability`：骨架出现频率（0~1）
  - `dir_stability`：方向一致频率（0~1，仅有向边）

使用建议：
- 小样本或噪声大：`--direction pacd`，适当减小 `--max-k`；
- 高维稀疏：`--direction s3cdo`，增加 `--s3cdo-top-m` 但注意计算成本；
- 多环境数据：设置 `--baseline-conds` 并提供 `--intervention`（若有先验）。

多环境数据（含 `COND`）使用说明：

- 脚本会自动将 `COND` one-hot 作为 **X 块**输入 IVAPCI，以吸收环境漂移；
- 若 `COND` 为字符串列，依然可直接读取并编码；
- 建议合成数据的基线环境设置为 `env_0`（例如 `--baseline-conds env_0`）。

当 `--direction pacd` 且数据包含 `COND` 时，脚本会像主流水线一样使用多环境干预证据：

- `--baseline-conds` 指定基线环境（默认 `CD3CD28,CD3CD28+ICAM2`）
- `--intervention` 指向干预映射 JSON（若不提供，会把非基线环境视为干预）

示例（合成数据）：

```bash
python run_direction_ivapci_pipeline.py \
  --data multienv_soft_high.csv \
  --output results/dir_ivapci \
  --direction pacd \
  --baseline-conds env_0
```

示例（自定义干预映射）：

```bash
python run_direction_ivapci_pipeline.py \
  --data sachs_data.csv \
  --output results/dir_ivapci \
  --direction pacd \
  --baseline-conds CD3CD28,CD3CD28+ICAM2 \
  --intervention sachs_intervention_map.json
```

输出：
- `edge_effects.csv`
- `edge_effects.json`
