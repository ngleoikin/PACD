"""
PACD 结构学习模块
=================

基于 PACD 理论实现:
1. p-adic 特征映射 (Section 2)
2. 因果方向识别 (Theorem 3.1) - 通过预测风险比较
3. 条件独立性检验 (Theorem 4.1) - p-adic 特征上的CI检验
4. 骨架学习 (PC算法 + p-adic CI)
5. 稳定性增强 (Bootstrap + 集成规则)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


@dataclass
class PACDStructureConfig:
    """PACD结构学习配置"""

    # p-adic 参数
    p: int = 2
    m: int = 4

    # CI检验参数
    alpha: float = 0.001
    max_k: int = 3

    # 方向识别参数
    direction_threshold: float = 0.1
    n_folds: int = 5

    # 稳定性与集成参数
    n_bootstrap: int = 20
    edge_freq_threshold: float = 0.6
    residual_dependence_weight: float = 0.5

    # 其他
    seed: int = 42


class PAdicFeatureMapper:
    """
    p-adic 特征映射 (Section 2)

    将连续变量映射到有限精度的 p-adic 表示:
    φ_{p,m}(x) = Σ_{k=1}^{m} a_k * p^{-k}
    """

    def __init__(self, p: int = 2, m: int = 4):
        self.p = p
        self.m = m

    def transform_scalar(self, x: float) -> float:
        """对单个标量进行 p-adic 截断"""
        x = np.clip(x, 0, 1)
        result = 0.0
        remaining = x

        for k in range(1, self.m + 1):
            scale = self.p ** (-k)
            a_k = int(remaining / scale)
            a_k = min(a_k, self.p - 1)
            result += a_k * scale
            remaining -= a_k * scale
            remaining = max(0, remaining)

        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """对数据矩阵进行 p-adic 变换"""
        X = np.asarray(X)
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_normalized = (X - X_min) / X_range
        transform_vec = np.vectorize(self.transform_scalar)
        return transform_vec(X_normalized)


class PAdicCITest:
    """基于 p-adic 特征的条件独立性检验 (Theorem 4.1)"""

    def __init__(self, config: PACDStructureConfig):
        self.config = config
        self.mapper = PAdicFeatureMapper(config.p, config.m)

    def test_partial_correlation(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        S: List[int],
    ) -> Tuple[bool, float, float]:
        """基于 p-adic 特征的偏相关检验"""
        n = X.shape[0]

        if not S:
            r, _ = spearmanr(X[:, i], X[:, j])
            z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(abs(z) / se))
            return p_value > self.config.alpha, r, p_value

        X_S = X[:, S]
        Phi_S = self.mapper.transform(X_S)
        model_i = Ridge(alpha=0.1)
        model_i.fit(Phi_S, X[:, i])
        res_i = X[:, i] - model_i.predict(Phi_S)

        model_j = Ridge(alpha=0.1)
        model_j.fit(Phi_S, X[:, j])
        res_j = X[:, j] - model_j.predict(Phi_S)

        r, _ = spearmanr(res_i, res_j)
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        df = max(n - len(S) - 3, 1)
        se = 1.0 / np.sqrt(df)
        p_value = 2 * (1 - norm.cdf(abs(z) / se))

        return p_value > self.config.alpha, r, p_value


class PAdicDirectionIdentifier:
    """基于 p-adic 特征的因果方向识别 (Theorem 3.1)"""

    def __init__(self, config: PACDStructureConfig):
        self.config = config
        self.mapper = PAdicFeatureMapper(config.p, config.m)

    def compute_prediction_risk(self, X: np.ndarray, Y: np.ndarray) -> float:
        """计算 E[|Y - h(Φ(X))|²] 的估计"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Phi_X = self.mapper.transform(X)
        kf = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed,
        )
        mse_scores = []
        for train_idx, test_idx in kf.split(Phi_X):
            model = Ridge(alpha=0.1)
            model.fit(Phi_X[train_idx], Y[train_idx])
            y_pred = model.predict(Phi_X[test_idx])
            mse_scores.append(mean_squared_error(Y[test_idx], y_pred))
        return float(np.mean(mse_scores))

    def compute_residual_dependence(self, X: np.ndarray, Y: np.ndarray) -> float:
        """估计残差与输入的相关性（越小越符合因果方向）"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Phi_X = self.mapper.transform(X)
        model = Ridge(alpha=0.1)
        model.fit(Phi_X, Y)
        residuals = Y - model.predict(Phi_X)
        corr, _ = spearmanr(residuals, X[:, 0])
        return float(abs(corr))

    def identify_direction(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        var_name_x: str = "X",
        var_name_y: str = "Y",
        residual_weight: float = 0.0,
    ) -> Dict:
        """识别 X 和 Y 之间的因果方向"""
        risk_xy = self.compute_prediction_risk(X, Y)
        risk_yx = self.compute_prediction_risk(Y, X)
        dep_xy = self.compute_residual_dependence(X, Y)
        dep_yx = self.compute_residual_dependence(Y, X)

        score_xy = risk_xy + residual_weight * dep_xy
        score_yx = risk_yx + residual_weight * dep_yx
        score_diff = score_yx - score_xy

        if score_diff > self.config.direction_threshold:
            direction = f"{var_name_x}->{var_name_y}"
            source, target = var_name_x, var_name_y
            confidence = min(abs(score_diff) / (score_xy + 1e-10), 1.0)
        elif score_diff < -self.config.direction_threshold:
            direction = f"{var_name_y}->{var_name_x}"
            source, target = var_name_y, var_name_x
            confidence = min(abs(score_diff) / (score_yx + 1e-10), 1.0)
        else:
            direction = f"{var_name_x}--{var_name_y}"
            source, target = var_name_x, var_name_y
            confidence = 0.0

        return {
            "direction": direction,
            "source": source,
            "target": target,
            "confidence": float(confidence),
            "risk_xy": float(risk_xy),
            "risk_yx": float(risk_yx),
            "risk_diff": float(risk_yx - risk_xy),
            "residual_dep_xy": float(dep_xy),
            "residual_dep_yx": float(dep_yx),
        }


class PACDStructureLearner:
    """PACD 结构学习器 (PC + p-adic CI)"""

    def __init__(self, config: Optional[PACDStructureConfig] = None):
        self.config = config or PACDStructureConfig()
        self.ci_tester = PAdicCITest(self.config)
        self.direction_identifier = PAdicDirectionIdentifier(self.config)

        self.skeleton_: Set[Tuple[int, int]] = set()
        self.sepsets_: Dict[Tuple[int, int], List[int]] = {}
        self.directions_: Dict[Tuple[int, int], Dict] = {}
        self.edge_pvalues_: Dict[Tuple[int, int], float] = {}

    def _neighbors(self, edges: Set[Tuple[int, int]], node: int, exclude: int) -> Set[int]:
        neighbors: Set[int] = set()
        for (a, b) in edges:
            if a == node and b != exclude:
                neighbors.add(b)
            elif b == node and a != exclude:
                neighbors.add(a)
        return neighbors

    def learn_skeleton(self, X: np.ndarray) -> Set[Tuple[int, int]]:
        """PC算法学习骨架 (使用 p-adic CI检验)"""
        _, d = X.shape
        edges = {(i, j) for i in range(d) for j in range(i + 1, d)}

        for k in range(self.config.max_k + 1):
            to_remove = []
            for (i, j) in list(edges):
                neighbors = self._neighbors(edges, i, j) | self._neighbors(edges, j, i)
                if len(neighbors) < k:
                    continue
                for S in combinations(neighbors, k):
                    is_indep, _, _ = self.ci_tester.test_partial_correlation(X, i, j, list(S))
                    if is_indep:
                        to_remove.append((i, j))
                        self.sepsets_[(i, j)] = list(S)
                        self.sepsets_[(j, i)] = list(S)
                        break
                if (i, j) not in to_remove:
                    _, _, p = self.ci_tester.test_partial_correlation(X, i, j, [])
                    self.edge_pvalues_[(i, j)] = p
            for e in to_remove:
                edges.discard(e)

        self.skeleton_ = edges
        return edges

    def orient_edges(self, X: np.ndarray, var_names: List[str]) -> List[Dict]:
        """使用 p-adic 方向识别定向边"""
        directed_edges = []
        for (i, j) in self.skeleton_:
            result = self.direction_identifier.identify_direction(
                X[:, i],
                X[:, j],
                var_names[i],
                var_names[j],
            )
            self.directions_[(i, j)] = result
            directed_edges.append(
                {
                    "source": result["source"],
                    "target": result["target"],
                    "direction_confidence": (
                        "high"
                        if result["confidence"] > 0.5
                        else ("medium" if result["confidence"] > 0.2 else "low")
                    ),
                    "risk_diff": result["risk_diff"],
                    "sepset": [var_names[s] for s in self.sepsets_.get((i, j), [])],
                }
            )
        return directed_edges

    def learn(self, X: np.ndarray, var_names: List[str]) -> Dict:
        """完整结构学习"""
        skeleton = self.learn_skeleton(X)
        directed = self.orient_edges(X, var_names)
        sepsets_named = {
            f"{var_names[i]}|{var_names[j]}": [var_names[s] for s in S]
            for (i, j), S in self.sepsets_.items()
        }
        return {
            "skeleton": [(var_names[i], var_names[j]) for (i, j) in skeleton],
            "skeleton_indices": list(skeleton),
            "directed_edges": directed,
            "sepsets": sepsets_named,
            "n_edges": len(directed),
        }


class PACDEnsembleStructureLearner(PACDStructureLearner):
    """
    集成式结构学习器:
    - Bootstrap 稳定性选择
    - 风险 + 残差依赖度联合定向
    """

    def _bootstrap_indices(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        return rng.integers(0, n, size=n)

    def learn_skeleton(self, X: np.ndarray) -> Set[Tuple[int, int]]:
        n, d = X.shape
        edge_counts = np.zeros((d, d), dtype=int)
        base_edges = set()

        for _ in range(self.config.n_bootstrap):
            idx = self._bootstrap_indices(n)
            learner = PACDStructureLearner(self.config)
            edges = learner.learn_skeleton(X[idx])
            base_edges |= edges
            for (i, j) in edges:
                edge_counts[i, j] += 1
                edge_counts[j, i] += 1

        threshold = int(np.ceil(self.config.edge_freq_threshold * self.config.n_bootstrap))
        stable_edges = {
            (i, j)
            for (i, j) in base_edges
            if edge_counts[i, j] >= threshold
        }

        self.skeleton_ = stable_edges
        return stable_edges

    def orient_edges(self, X: np.ndarray, var_names: List[str]) -> List[Dict]:
        directed_edges = []
        for (i, j) in self.skeleton_:
            result = self.direction_identifier.identify_direction(
                X[:, i],
                X[:, j],
                var_names[i],
                var_names[j],
                residual_weight=self.config.residual_dependence_weight,
            )
            self.directions_[(i, j)] = result
            directed_edges.append(
                {
                    "source": result["source"],
                    "target": result["target"],
                    "direction_confidence": (
                        "high"
                        if result["confidence"] > 0.5
                        else ("medium" if result["confidence"] > 0.2 else "low")
                    ),
                    "risk_diff": result["risk_diff"],
                    "residual_dep_xy": result["residual_dep_xy"],
                    "residual_dep_yx": result["residual_dep_yx"],
                }
            )
        return directed_edges


def demo() -> None:
    np.random.seed(42)
    n = 1000
    X1 = np.random.randn(n)
    X2 = 0.8 * X1 + np.random.randn(n) * 0.5
    X3 = 0.5 * X1 + 0.6 * X2 + np.random.randn(n) * 0.3

    data = np.column_stack([X1, X2, X3])
    var_names = ["X1", "X2", "X3"]

    config = PACDStructureConfig(p=2, m=4, alpha=0.01)
    learner = PACDEnsembleStructureLearner(config)
    result = learner.learn(data, var_names)

    print("\n结果:")
    print(f"  边数: {result['n_edges']}")
    for edge in result["directed_edges"]:
        print(
            f"  {edge['source']} -> {edge['target']} "
            f"(conf={edge['direction_confidence']})"
        )


if __name__ == "__main__":
    demo()
