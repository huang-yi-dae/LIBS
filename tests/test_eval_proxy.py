"""eval_proxy.py 纯函数的秒级单元测试（不加载数据、不训练模型）。"""

import numpy as np
import pytest

from eval_proxy import _spearman


class TestSpearman:
    def test_perfect_positive(self):
        assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_rank_based_not_value_based(self):
        # 单调但非线性 → 秩相关仍为 +1（区别于 Pearson）
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 100.0, 101.0, 10000.0]
        assert _spearman(a, b) == pytest.approx(1.0)

    def test_known_value(self):
        # 交换一对相邻秩：4 点 Spearman = 1 - 6*Σd²/(n(n²-1)) = 1 - 6*2/60 = 0.8
        assert _spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)

    def test_single_point_returns_nan(self):
        # 单点序列秩方差为 0 → 分母为 0 → nan
        assert np.isnan(_spearman([1.0], [2.0]))

    def test_ties_get_distinct_ranks(self):
        # 实现用双重 argsort 排秩：并列值按出现顺序拿到不同秩（非平均秩），
        # 常数序列因此不会触发 nan 分支——锁定该行为防止静默变更
        assert _spearman([1, 1, 1], [1, 2, 3]) == pytest.approx(1.0)


class TestGlobalWeightedRmse:
    """AGENTS.md 口径: global = sqrt(Σ(rmse_i² × n_test_i) / Σ(n_test_i))。

    eval_proxy.run_config / eval_combos.global_at 均用
    sqrt(np.average(r**2, weights=w)) 实现该公式，此处锁定两者等价。
    """

    def test_matches_manual_formula(self):
        r = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        w = np.array([5, 26, 9], dtype=np.float32)
        got = float(np.sqrt(np.average(r ** 2, weights=w)))
        expected = float(np.sqrt((r ** 2 * w).sum() / w.sum()))
        assert got == pytest.approx(expected)

    def test_equal_weights_reduces_to_rms(self):
        r = np.array([3.0, 4.0])
        w = np.array([1.0, 1.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        got = float(np.sqrt(np.average(r ** 2, weights=w)))
        assert got == pytest.approx(np.sqrt(12.5))

    def test_weighting_pulls_toward_heavy_coal(self):
        r = np.array([100.0, 300.0])
        light = float(np.sqrt(np.average(r ** 2, weights=[9.0, 1.0])))
        heavy = float(np.sqrt(np.average(r ** 2, weights=[1.0, 9.0])))
        assert light < heavy  # 权重偏向高 RMSE 煤种 → 全局更大
