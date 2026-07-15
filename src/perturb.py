"""
光谱扰动模块 — 模拟仪器偏差

提供两类扰动，均在 (波长, 强度) 原始配对层面操作：

  1. 波长偏移 (wavelength shift)
     模拟光谱仪波长校准漂移。将强度插值到偏移后的波长网格上，
     然后再映射回原始波长网格（重采样）。

  2. 基线偏移 (baseline shift)
     模拟背景光/连续谱/基线漂移。在原始强度上叠加不同类型的人造基线。
"""

import numpy as np
from typing import Callable, Optional


# ── 波长偏移 ──────────────────────────────────────────────────────────────────

def apply_wave_shift(wl: np.ndarray, inten: np.ndarray,
                     shift_nm: float) -> np.ndarray:
    """
    对单条光谱施加波长偏移。

    原理: 光谱仪若存在校准误差，实际采集的波长对应关系为
          λ_measured = λ_true + shift_nm
    我们将强度从偏移后的位置重采样回原始波长网格。

    参数:
        wl:       原始波长数组 (N,)
        inten:    原始强度数组 (N,)
        shift_nm: 偏移量 (nm)。正值=波长向长波方向偏移(红移)，
                  负值=向短波方向偏移(蓝移)

    返回:
        偏移后的强度数组 (N,)，与 wl 对应
    """
    if abs(shift_nm) < 1e-6:
        return inten.copy()

    # 偏移后的"观察波长": 光谱仪认为的波长与实际波长的关系
    # 如果 shift_nm > 0, 意味着实际波长比显示波长小 shift_nm
    # 所以我们在 wl_shifted = wl - shift_nm 处采样强度
    wl_shifted = wl - shift_nm

    # 线性插值回原始波长网格
    return np.interp(wl, wl_shifted, inten, left=0.0, right=0.0).astype(np.float32)


# ── 基线偏移 ──────────────────────────────────────────────────────────────────

def _constant_baseline(wl: np.ndarray, amp: float) -> np.ndarray:
    """常数基线偏移"""
    return np.full_like(wl, amp, dtype=np.float32)


def _linear_baseline(wl: np.ndarray, slope: float, intercept: float = 0.0) -> np.ndarray:
    """线性漂移基线"""
    return (slope * wl + intercept).astype(np.float32)


def _polynomial_baseline(wl: np.ndarray, coeffs: list) -> np.ndarray:
    """多项式基线: coeffs = [c0, c1, c2, ...] 对应常数项到高次项"""
    base = np.zeros_like(wl, dtype=np.float64)
    for i, c in enumerate(coeffs):
        base += c * (wl ** i)
    return base.astype(np.float32)


def _sinusoidal_baseline(wl: np.ndarray, amp: float,
                         freq: float, phase: float = 0.0) -> np.ndarray:
    """正弦基线 — 模拟周期性背景光波动"""
    return (amp * np.sin(2 * np.pi * freq * wl + phase)).astype(np.float32)


BASELINE_FACTORIES: dict[str, Callable] = {
    "constant":    _constant_baseline,
    "linear":      _linear_baseline,
    "polynomial":  _polynomial_baseline,
    "sine":        _sinusoidal_baseline,
}


def apply_baseline_shift(wl: np.ndarray, inten: np.ndarray,
                         shift_type: str = "constant",
                         **kwargs) -> np.ndarray:
    """
    对单条光谱叠加人造基线。

    参数:
        wl:          波长数组 (N,)
        inten:       原始强度数组 (N,)
        shift_type:  基线类型 ("constant", "linear", "polynomial", "sine")
        **kwargs:    传给具体基线生成函数的参数

    返回:
        叠加基线后的强度数组 (N,)
    """
    if shift_type == "none" or shift_type is None:
        return inten.copy()

    factory = BASELINE_FACTORIES.get(shift_type)
    if factory is None:
        raise ValueError(f"未知基线类型: {shift_type}，可选: {list(BASELINE_FACTORIES.keys())}")

    baseline = factory(wl, **kwargs)
    result = inten + baseline
    return np.maximum(result, 0.0).astype(np.float32)


# ── 批处理 ────────────────────────────────────────────────────────────────────

def perturb_spectrum(wl: np.ndarray, inten: np.ndarray,
                     wave_shift: float = 0.0,
                     baseline_type: str = "none",
                     baseline_kw: Optional[dict] = None) -> np.ndarray:
    """
    对单条光谱依次施加波长偏移 → 基线偏移。

    参数:
        wl:            波长数组
        inten:         强度数组
        wave_shift:    波长偏移量 (nm)，0=不偏移
        baseline_type: 基线类型，"none"=不叠加
        baseline_kw:   基线参数字典

    返回:
        扰动后的强度数组
    """
    inten_out = inten.copy()

    # 先波长偏移
    if abs(wave_shift) > 1e-6:
        inten_out = apply_wave_shift(wl, inten_out, wave_shift)

    # 再基线偏移
    if baseline_type is not None and baseline_type != "none":
        kwargs = baseline_kw or {}
        inten_out = apply_baseline_shift(wl, inten_out, baseline_type, **kwargs)

    return inten_out
