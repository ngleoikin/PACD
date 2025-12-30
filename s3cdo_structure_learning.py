"""
S3C-DO 结构学习模块
===================

实现“筛选-清洗-定向”的结构学习框架：
1) Screen: 相关性筛选构造候选图
2) Clean: 局部化 PC-stable 清洗得到骨架
3) Orient: v-structure + Meek 规则定向 CPDAG
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import Ridge


@dataclass
class S3CDOConfig:
    """S3C-DO 配置"""

    # Screen
    top_m: int = 8
    screen_method: str = "spearman"  # spearman / pearson

    # Clean (PC-stable)
    alpha: float = 0.001
    max_k: int = 3
    ci_method: str = "spearman"  # spearman / pearson
    use_nonparanormal: bool = False
    ridge_alpha: float = 0.1

    # Orient
    orient_vstructures: bool = True
    use_meek_rules: bool = True
    ensure_acyclic: bool = True


class S3CDOStructureLearner:
    """S3C-DO 结构学习器"""

    def __init__(self, config: Optional[S3CDOConfig] = None):
        self.config = config or S3CDOConfig()
        self.skeleton_: Set[Tuple[int, int]] = set()
        self.sepsets_: Dict[Tuple[int, int], List[int]] = {}
        self.edge_scores_: Dict[Tuple[int, int], float] = {}

    def _nonparanormal_transform(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        X_transformed = np.zeros_like(X, dtype=float)
        for j in range(d):
            ranks = np.argsort(np.argsort(X[:, j])) + 1
            delta = 1.0 / (4.0 * n**0.25)
            ranks = np.clip(ranks / (n + 1), delta, 1 - delta)
            X_transformed[:, j] = norm.ppf(ranks)
        return X_transformed

    def _corr_score(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.config.screen_method == "pearson":
            r, _ = pearsonr(x, y)
        else:
            r, _ = spearmanr(x, y)
        return float(abs(r))

    def _screen(self, X: np.ndarray) -> Set[Tuple[int, int]]:
        n, d = X.shape
        scores = np.zeros((d, d), dtype=float)
        for i in range(d):
            for j in range(i + 1, d):
                scores[i, j] = self._corr_score(X[:, i], X[:, j])
                scores[j, i] = scores[i, j]
        edges: Set[Tuple[int, int]] = set()
        for i in range(d):
            idx = np.argsort(scores[i])[::-1]
            chosen = [j for j in idx if j != i][: self.config.top_m]
            for j in chosen:
                u, v = (i, j) if i < j else (j, i)
                edges.add((u, v))
                self.edge_scores_[(u, v)] = float(scores[i, j])
        return edges

    def _partial_corr_test(
        self, X: np.ndarray, i: int, j: int, S: List[int]
    ) -> Tuple[bool, float, float]:
        n = X.shape[0]
        if not S:
            if self.config.ci_method == "pearson":
                r, _ = pearsonr(X[:, i], X[:, j])
            else:
                r, _ = spearmanr(X[:, i], X[:, j])
            z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(abs(z) / se))
            return p_value > self.config.alpha, r, p_value

        X_S = X[:, S]
        model_i = Ridge(alpha=self.config.ridge_alpha).fit(X_S, X[:, i])
        res_i = X[:, i] - model_i.predict(X_S)
        model_j = Ridge(alpha=self.config.ridge_alpha).fit(X_S, X[:, j])
        res_j = X[:, j] - model_j.predict(X_S)

        if self.config.ci_method == "pearson":
            r, _ = pearsonr(res_i, res_j)
        else:
            r, _ = spearmanr(res_i, res_j)
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        df = max(n - len(S) - 3, 1)
        se = 1.0 / np.sqrt(df)
        p_value = 2 * (1 - norm.cdf(abs(z) / se))
        return p_value > self.config.alpha, r, p_value

    def _neighbors(self, edges: Set[Tuple[int, int]], node: int, exclude: int) -> Set[int]:
        neighbors: Set[int] = set()
        for (a, b) in edges:
            if a == node and b != exclude:
                neighbors.add(b)
            elif b == node and a != exclude:
                neighbors.add(a)
        return neighbors

    def _is_adjacent(
        self, a: int, b: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> bool:
        if a == b:
            return True
        u, v = (a, b) if a < b else (b, a)
        if (u, v) in undirected:
            return True
        return (a, b) in directed or (b, a) in directed

    def _would_create_cycle(self, directed: Set[Tuple[int, int]], src: int, dst: int) -> bool:
        stack = [dst]
        visited = set()
        adj: Dict[int, Set[int]] = {}
        for a, b in directed:
            adj.setdefault(a, set()).add(b)
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj.get(cur, ()):
                if nxt not in visited:
                    stack.append(nxt)
        return False

    def _orient_edge(
        self, src: int, dst: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> bool:
        if (dst, src) in directed:
            return False
        if self.config.ensure_acyclic and self._would_create_cycle(directed, src, dst):
            return False
        u, v = (src, dst) if src < dst else (dst, src)
        undirected.discard((u, v))
        if (src, dst) not in directed:
            directed.add((src, dst))
            return True
        return False

    def _orient_vstructures(
        self, d: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> None:
        if not self.config.orient_vstructures:
            return
        for k in range(d):
            nbrs = [n for n in range(d) if n != k and self._is_adjacent(k, n, undirected, directed)]
            for i, j in combinations(nbrs, 2):
                if self._is_adjacent(i, j, undirected, directed):
                    continue
                sepset = self.sepsets_.get((i, j)) or self.sepsets_.get((j, i)) or []
                if k not in sepset:
                    self._orient_edge(i, k, undirected, directed)
                    self._orient_edge(j, k, undirected, directed)

    def _apply_meek_rules(
        self, d: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> None:
        if not self.config.use_meek_rules:
            return
        changed = True
        while changed:
            changed = False
            dir_list = list(directed)
            und_list = list(undirected)
            for a, b in dir_list:
                for (u, v) in und_list:
                    if u == b:
                        c = v
                    elif v == b:
                        c = u
                    else:
                        continue
                    if c == a:
                        continue
                    if not self._is_adjacent(a, c, undirected, directed):
                        if self._orient_edge(b, c, undirected, directed):
                            changed = True

            und_list = list(undirected)
            for (u, v) in und_list:
                for c in range(d):
                    if (u, c) in directed and (c, v) in directed:
                        if self._orient_edge(u, v, undirected, directed):
                            changed = True
                        break
                    if (v, c) in directed and (c, u) in directed:
                        if self._orient_edge(v, u, undirected, directed):
                            changed = True
                        break

    def learn(self, X: np.ndarray, var_names: List[str]) -> Dict:
        X = np.asarray(X)
        if self.config.use_nonparanormal:
            X = self._nonparanormal_transform(X)

        candidate_edges = self._screen(X)
        edges = set(candidate_edges)

        for k in range(self.config.max_k + 1):
            to_remove = []
            for (i, j) in list(edges):
                neighbors = self._neighbors(edges, i, j) | self._neighbors(edges, j, i)
                if len(neighbors) < k:
                    continue
                for S in combinations(neighbors, k):
                    is_indep, _, _ = self._partial_corr_test(X, i, j, list(S))
                    if is_indep:
                        to_remove.append((i, j))
                        self.sepsets_[(i, j)] = list(S)
                        self.sepsets_[(j, i)] = list(S)
                        break
            for e in to_remove:
                edges.discard(e)

        self.skeleton_ = edges
        undirected = {tuple(sorted(e)) for e in edges}
        directed: Set[Tuple[int, int]] = set()

        self._orient_vstructures(X.shape[1], undirected, directed)
        self._apply_meek_rules(X.shape[1], undirected, directed)

        directed_edges: List[Dict] = []
        for (src, dst) in sorted(directed):
            sepset_idx = self.sepsets_.get((src, dst)) or self.sepsets_.get((dst, src)) or []
            directed_edges.append(
                {
                    "source": var_names[src],
                    "target": var_names[dst],
                    "orientation_method": "rule",
                    "direction_confidence": "rule",
                    "sepset": [var_names[s] for s in sepset_idx],
                }
            )

        for (u, v) in sorted(undirected):
            sepset_idx = self.sepsets_.get((u, v)) or self.sepsets_.get((v, u)) or []
            directed_edges.append(
                {
                    "source": var_names[u],
                    "target": var_names[v],
                    "orientation_method": "undirected",
                    "direction_confidence": "undecided",
                    "sepset": [var_names[s] for s in sepset_idx],
                }
            )

        return {
            "candidate_edges": [(var_names[i], var_names[j]) for (i, j) in candidate_edges],
            "skeleton": [(var_names[i], var_names[j]) for (i, j) in edges],
            "skeleton_indices": list(edges),
            "directed_edges": directed_edges,
            "sepsets": {
                f"{var_names[i]}|{var_names[j]}": [var_names[s] for s in S]
                for (i, j), S in self.sepsets_.items()
            },
        }

