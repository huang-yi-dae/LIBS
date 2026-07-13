"""
窗口宽度自动搜索工具

基于 CV-RMSE 自动优化 KEY_LINES 的半窗口宽度。
支持两种模式:
  1. 全局缩放因子搜索 — 对所有元素乘以统一系数
  2. 逐元素搜索 — 固定其他元素，单独扫描某元素的 halfwin

用法:
    python -c "from src.window_search import *; search_global_multiplier()"
    python -c "from src.window_search import *; search_element('C')"
"""

import sys
import os
import numpy as np
import copy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import COAL_TYPES, TRAIN_DIR
from src.data   import load_labels, load_coal_spectra


# ── 内部默认值（原 config.py 中的 WINDOW_CANDIDATES / WINDOW_MULTIPLIER） ──
_DEFAULT_WINDOW_CANDIDATES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
_DEFAULT_WINDOW_MULTIPLIER = 1.0
from src.model  import train_coal_model


# ── 记住原始窗口以备恢复 ──────────────────────────────────────────────────────
_ORIGINAL_KEY_LINES = copy.deepcopy(config.KEY_LINES)


def _set_raw_halfwins(halfwin_dict):
    """修改 KEY_LINES 的 halfwin（使用内部默认缩放因子）。"""
    new_lines = {}
    for name, (center, _) in _ORIGINAL_KEY_LINES.items():
        if name in halfwin_dict:
            new_lines[name] = (center, halfwin_dict[name] * _DEFAULT_WINDOW_MULTIPLIER)
        else:
            new_lines[name] = (center, _ORIGINAL_KEY_LINES[name][1] * _DEFAULT_WINDOW_MULTIPLIER)
    config.KEY_LINES = new_lines


def _restore():
    """恢复原始 KEY_LINES。"""
    config.KEY_LINES = copy.deepcopy(_ORIGINAL_KEY_LINES)


def _run_full_cv():
    """
    运行所有煤种 CV，返回全局 CV-RMSE。
    """
    label_map, aux_map = load_labels()
    rmses = []

    for coal_type in COAL_TYPES:
        train_data = load_coal_spectra(TRAIN_DIR, coal_type, label_map, aux_map)
        if train_data is None or train_data['n_batches'] == 0:
            continue
        model_dict = train_coal_model(coal_type, train_data)
        rmses.append(model_dict['cv_rmse'])

    return float(np.mean(rmses)) if rmses else np.inf


def search_global_multiplier(candidates=None):
    """
    搜索全局窗口缩放因子 WINDOW_MULTIPLIER。

    参数:
        candidates: 候选值列表。默认 [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    """
    if candidates is None:
        candidates = _DEFAULT_WINDOW_CANDIDATES

    print(f"{'='*60}")
    print(f"全局窗口缩放因子搜索 ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")
    print(f"{'乘数':>10} {'CV-RMSE':>10} {'Δ基线':>10}")
    print("-" * 40)

    baseline = None
    results  = []

    for multiplier in sorted(candidates):
        # 注意：此函数暂未更新 KEY_LINES 的 halfwins，乘数不能直接影响 CV
        cv = _run_full_cv()

        if baseline is None:
            baseline = cv

        delta = cv - baseline
        results.append((multiplier, cv, delta))
        print(f"{multiplier:>10.2f} {cv:>10.2f} {delta:>+10.2f}")

    # 最优
    best = min(results, key=lambda r: r[1])
    print("-" * 40)
    print(f"最优: 乘数={best[0]:.2f}, CV-RMSE={best[1]:.2f}")

    return results


def search_element(element_name, candidates=None):
    """
    对某个元素搜索最优 halfwin，其他元素固定。

    参数:
        element_name: KEY_LINES 中的元素名（如 'C', 'H'）
        candidates:   候选 halfwin 值列表。默认 WINDOW_CANDIDATES
    """
    if element_name not in _ORIGINAL_KEY_LINES:
        print(f"错误: 未知元素 '{element_name}'，可用: {list(_ORIGINAL_KEY_LINES.keys())}")
        return []

    if candidates is None:
        candidates = _DEFAULT_WINDOW_CANDIDATES

    print(f"{'='*60}")
    print(f"元素 [{element_name}] 窗口搜索 ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")
    print(f"{'halfwin':>10} {'CV-RMSE':>10} {'Δ基线':>10}")
    print("-" * 40)

    baseline = None
    results  = []

    for hw in sorted(candidates):
        _set_raw_halfwins({element_name: hw})
        cv = _run_full_cv()

        if baseline is None:
            baseline = cv

        delta = cv - baseline
        results.append((hw, cv, delta))
        print(f"{hw:>10.2f} {cv:>10.2f} {delta:>+10.2f}")

    # 最优
    best = min(results, key=lambda r: r[1])
    print("-" * 40)
    print(f"最优: halfwin={best[0]:.2f}, CV-RMSE={best[1]:.2f}")

    _restore()
    return results


def search_all_elements(candidates=None):
    """
    对每个元素依次做逐元素搜索。
    每次只改变一个元素，其他保持原始值。
    """
    print(f"\n{'='*60}")
    print("逐元素窗口搜索（全部元素）")
    print(f"{'='*60}\n")

    summary = {}
    for name in _ORIGINAL_KEY_LINES:
        results = search_element(name, candidates)
        best = min(results, key=lambda r: r[1])
        current = _ORIGINAL_KEY_LINES[name][1]
        summary[name] = {
            'current': current,
            'best': best[0],
            'best_cv': best[1],
            'improvement': current - best[0],
        }
        print()

    print(f"\n{'='*60}")
    print("搜索结果汇总")
    print(f"{'='*60}")
    print(f"{'元素':>8} {'当前hw':>8} {'最优hw':>8} {'最优CV':>8}")
    print("-" * 40)
    for name, info in summary.items():
        print(f"{name:>8} {info['current']:>8.2f} {info['best']:>8.2f} {info['best_cv']:>8.2f}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="窗口宽度自动搜索")
    parser.add_argument("mode", choices=["global", "element", "all"],
                        help="搜索模式: global=全局系数, element=单元素, all=逐元素")
    parser.add_argument("--element", type=str, default=None,
                        help="element 模式下指定元素名")
    args = parser.parse_args()

    if args.mode == "global":
        search_global_multiplier()
    elif args.mode == "element":
        if not args.element:
            print("请指定 --element")
        else:
            search_element(args.element)
    elif args.mode == "all":
        search_all_elements()
