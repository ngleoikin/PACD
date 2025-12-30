"""
PACD-IVAPCI 统一包装模块
========================

支持调用:
- models/pacdt_v30.py (PACD-T v3.0)
- models/ivapci_v33_theory.py (IVAPCI v3.3)

使用方法:
    from model_wrapper import (
        check_pacd, check_ivapci,
        create_pacd_estimator, create_ivapci_estimator,
        estimate_ate_pacd, estimate_ate_ivapci
    )
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np


class BaseCausalEstimator:
    """Mock base class"""

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_ate(self, *args, **kwargs):
        raise NotImplementedError


_pacd_estimator = None
_pacd_config = None
_pacd_error = None
_pacd_initialized = False

_ivapci_estimator = None
_ivapci_config = None
_ivapci_error = None
_ivapci_initialized = False


def _find_module(possible_names: List[str]) -> Optional[Path]:
    """在常见位置搜索模块文件"""
    cwd = Path.cwd().resolve()

    search_dirs = [
        cwd / "models",
        cwd,
        Path(__file__).parent / "models",
        Path(__file__).parent,
    ]

    for directory in search_dirs:
        for name in possible_names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _load_module_with_mock(filepath: Path, module_name: str):
    """加载模块，处理相对导入"""
    import types

    with open(filepath, "r", encoding="utf-8") as handle:
        source = handle.read()

    pkg_name = f"{module_name}_mock_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.BaseCausalEstimator = BaseCausalEstimator
        pkg.__path__ = [str(filepath.parent)]
        sys.modules[pkg_name] = pkg

    modified = source.replace(
        "from . import BaseCausalEstimator",
        f"from {pkg_name} import BaseCausalEstimator",
    )

    mod = types.ModuleType(module_name)
    mod.__file__ = str(filepath)
    exec(compile(modified, str(filepath), "exec"), mod.__dict__)
    sys.modules[module_name] = mod

    return mod


def load_pacd():
    """加载PACD模块"""
    global _pacd_estimator, _pacd_config, _pacd_error, _pacd_initialized

    if _pacd_initialized:
        return _pacd_estimator, _pacd_config, _pacd_error

    _pacd_initialized = True

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from models.pacdt_v30 import PACDTv30Estimator, PACDTConfig
        _pacd_estimator = PACDTv30Estimator
        _pacd_config = PACDTConfig
        return _pacd_estimator, _pacd_config, None
    except Exception as exc:
        _pacd_error = f"直接import失败: {exc}"

    possible_names = [
        "pacdt_v30.py",
        "pacdt_v30_3.py",
        "pacdt.py",
        "pacd.py",
        "pacd_t.py",
    ]

    preferred = Path.cwd() / "models" / "pacdt_v30.py"
    if not preferred.exists():
        preferred = Path(__file__).parent / "models" / "pacdt_v30.py"
    filepath = preferred if preferred.exists() else _find_module(possible_names)

    if filepath is None:
        _pacd_error = f"找不到PACD文件，已搜索: {possible_names}"
        return None, None, _pacd_error

    print(f"  ✓ 找到PACD: {filepath}")

    try:
        mod = _load_module_with_mock(filepath, "pacd_loaded")
        _pacd_estimator = mod.PACDTv30Estimator
        _pacd_config = mod.PACDTConfig
        return _pacd_estimator, _pacd_config, None
    except Exception as exc:
        _pacd_error = f"加载PACD失败: {exc}"
        return None, None, _pacd_error


def load_ivapci():
    """加载IVAPCI模块"""
    global _ivapci_estimator, _ivapci_config, _ivapci_error, _ivapci_initialized

    if _ivapci_initialized:
        return _ivapci_estimator, _ivapci_config, _ivapci_error

    _ivapci_initialized = True

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from models.ivapci_v33_theory import (
            IVAPCIv33TheoryHierEstimator,
            IVAPCIV33TheoryConfig,
        )
        _ivapci_estimator = IVAPCIv33TheoryHierEstimator
        _ivapci_config = IVAPCIV33TheoryConfig
        return _ivapci_estimator, _ivapci_config, None
    except Exception as exc:
        _ivapci_error = f"直接import失败: {exc}"

    possible_names = [
        "ivapci_v33_theory.py",
        "ivapci_v33_theory_v9_0_stable.py",
        "ivapci.py",
    ]

    search_dirs = [
        Path.cwd() / "models",
        Path.cwd(),
        Path.cwd().parent / "models",
        Path(__file__).parent / "models",
        Path(__file__).parent,
    ]
    filepath = None
    for directory in search_dirs:
        for name in possible_names:
            candidate = directory / name
            if candidate.exists():
                filepath = candidate
                break
        if filepath is not None:
            break

    if filepath is None:
        searched = ", ".join(str(p) for p in search_dirs)
        _ivapci_error = (
            f"找不到IVAPCI文件，已搜索: {possible_names} | dirs: {searched}"
        )
        return None, None, _ivapci_error

    print(f"  ✓ 找到IVAPCI: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            source = handle.read()

        import types

        pkg_name = "ivapci_mock_pkg"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.BaseCausalEstimator = BaseCausalEstimator
            pkg.__path__ = [str(filepath.parent)]
            sys.modules[pkg_name] = pkg

        modified = source.replace(
            "from . import BaseCausalEstimator",
            f"from {pkg_name} import BaseCausalEstimator",
        )

        if "from .ivapci_theory_diagnostics import" in modified:
            mock_code = """
# Mock diagnostics
class TheoremComplianceDiagnostics:
    def __init__(self, *a, **kw): pass
    def update(self, *a, **kw): pass
    def get_summary(self): return {}
class TheoremDiagnosticsConfig: pass
"""
            import re
            pattern = r"from\\s+\\.\\s*ivapci_theory_diagnostics\\s+import\\s*\\([\\s\\S]*?\\)\\s*"
            modified, count = re.subn(pattern, mock_code + "\n", modified, count=1)
            if count == 0:
                modified = modified.replace(
                    "from .ivapci_theory_diagnostics import",
                    mock_code + "\n# from .ivapci_theory_diagnostics import",
                )

        mod = types.ModuleType("ivapci_loaded")
        mod.__file__ = str(filepath)
        mod.__package__ = pkg_name
        exec(compile(modified, str(filepath), "exec"), mod.__dict__)
        sys.modules["ivapci_loaded"] = mod

        _ivapci_estimator = mod.IVAPCIv33TheoryHierEstimator
        _ivapci_config = mod.IVAPCIV33TheoryConfig
        return _ivapci_estimator, _ivapci_config, None

    except SyntaxError as exc:
        location = f"{exc.filename}:{exc.lineno}"
        line = (exc.text or "").rstrip()
        _ivapci_error = f"加载IVAPCI失败: {exc.msg} ({location}) -> {line}"
        return None, None, _ivapci_error
    except Exception as exc:
        _ivapci_error = f"加载IVAPCI失败: {exc}"
        return None, None, _ivapci_error


def check_pacd() -> bool:
    """检查PACD是否可用"""
    est, _, err = load_pacd()
    if est is not None:
        print("✓ PACD 可用")
        return True
    print(f"✗ PACD 不可用: {err}")
    return False


def check_ivapci() -> bool:
    """检查IVAPCI是否可用"""
    est, _, err = load_ivapci()
    if est is not None:
        print("✓ IVAPCI 可用")
        return True
    print(f"✗ IVAPCI 不可用: {err}")
    return False


def is_pacd_available() -> bool:
    load_pacd()
    return _pacd_estimator is not None


def is_ivapci_available() -> bool:
    load_ivapci()
    return _ivapci_estimator is not None


def create_pacd_estimator(
    latent_dim_c: int = 4,
    latent_dim_n: int = 4,
    epochs: int = 200,
    device: str = "cpu",
):
    """创建PACD估计器"""
    est, cfg, err = load_pacd()
    if est is None:
        raise RuntimeError(f"PACD不可用: {err}")

    config = cfg(
        latent_dim_c=latent_dim_c,
        latent_dim_n=latent_dim_n,
        epochs=epochs,
        device=device,
    )
    return est(config)


def estimate_ate_pacd(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    epochs: int = 200,
    device: str = "cpu",
    n_bootstrap: int = 100,
) -> Dict:
    """使用PACD估计ATE"""
    from scipy.stats import norm

    n = len(A)
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    estimator = create_pacd_estimator(epochs=epochs, device=device)
    estimator.fit(X, A, Y)
    ate = estimator.estimate_ate(X, A, Y)

    ates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            est_b = create_pacd_estimator(
                epochs=max(50, epochs // 2), device=device
            )
            est_b.fit(X[idx], A[idx], Y[idx])
            ates.append(est_b.estimate_ate(X[idx], A[idx], Y[idx]))
        except Exception:
            ates.append(ate)

    se = np.std(ates)
    ci = [np.percentile(ates, 2.5), np.percentile(ates, 97.5)]
    p_value = 2 * (1 - norm.cdf(abs(ate) / (se + 1e-10)))

    return {
        "ate": float(ate),
        "se": float(se),
        "ci": [float(ci[0]), float(ci[1])],
        "p_value": float(p_value),
        "method": "pacd",
    }


def create_ivapci_estimator(
    x_dim: int,
    w_dim: int,
    z_dim: int,
    n_samples: int,
    epochs: int = 80,
    device: str = "cpu",
):
    """创建IVAPCI估计器"""
    est, cfg, err = load_ivapci()
    if est is None:
        raise RuntimeError(f"IVAPCI不可用: {err}")

    config = cfg(
        x_dim=x_dim,
        w_dim=w_dim,
        z_dim=z_dim,
        n_samples_hint=n_samples,
        epochs_main=epochs,
        epochs_pretrain=min(30, epochs // 3),
        device=device,
        seed=42,
    )
    return est(config)


def estimate_ate_ivapci(
    V_all: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    x_dim: int,
    w_dim: int,
    z_dim: int,
    epochs: int = 80,
    device: str = "cpu",
    n_bootstrap: int = 100,
    progress: Optional[Dict[str, int]] = None,
) -> Dict:
    """使用IVAPCI估计ATE"""
    from scipy.stats import norm

    n = len(A)
    V_all = np.asarray(V_all, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if progress:
        scenario = progress.get("base_scenario", 1)
        total = progress.get("total_scenarios", 1)
        edge_index = progress.get("edge_index", 1)
        edge_total = progress.get("edge_total", 1)
        print(
            f"[IVAPCI] scenario {scenario}/{total} "
            f"(edge {edge_index}/{edge_total}, main)"
        )
    estimator = create_ivapci_estimator(x_dim, w_dim, z_dim, n, epochs, device)
    estimator.fit(V_all, A, Y)
    ate = estimator.estimate_ate(V_all, A, Y)
    diagnostics = getattr(estimator, "training_diagnostics", {})

    ates = []
    for b in range(n_bootstrap):
        if progress:
            scenario = progress.get("base_scenario", 1) + b + 1
            total = progress.get("total_scenarios", 1)
            edge_index = progress.get("edge_index", 1)
            edge_total = progress.get("edge_total", 1)
            print(
                f"[IVAPCI] scenario {scenario}/{total} "
                f"(edge {edge_index}/{edge_total}, bootstrap {b + 1}/{n_bootstrap})"
            )
        idx = np.random.choice(n, n, replace=True)
        try:
            est_b = create_ivapci_estimator(
                x_dim,
                w_dim,
                z_dim,
                n,
                max(30, epochs // 2),
                device,
            )
            est_b.fit(V_all[idx], A[idx], Y[idx])
            ates.append(est_b.estimate_ate(V_all[idx], A[idx], Y[idx]))
        except Exception:
            ates.append(ate)

    se = np.std(ates)
    ci = [np.percentile(ates, 2.5), np.percentile(ates, 97.5)]
    p_value = 2 * (1 - norm.cdf(abs(ate) / (se + 1e-10)))

    return {
        "ate": float(ate),
        "se": float(se),
        "ci": [float(ci[0]), float(ci[1])],
        "p_value": float(p_value),
        "method": "ivapci",
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("模型包装器测试")
    print("=" * 50)
    print(f"工作目录: {Path.cwd()}")
    print()

    print("检查PACD...")
    check_pacd()

    print("\n检查IVAPCI...")
    check_ivapci()

    print("\n期望的目录结构:")
    print("  your_project/")
    print("  ├── models/")
    print("  │   ├── pacdt_v30.py       # PACD-T")
    print("  │   └── ivapci_v33_theory.py  # IVAPCI")
    print("  └── model_wrapper.py")
