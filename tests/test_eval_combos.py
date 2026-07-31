"""eval_combos.py 纯函数（_agg 批次聚合 / coal_rmse 收缩）的秒级单元测试。"""

import numpy as np
import pytest

from eval_combos import _agg, coal_rmse


class TestAgg:
    # 10 个点，含一高一低离群值，用于区分各聚合器
    VALS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]

    def test_default_is_median(self):
        assert _agg(self.VALS, "median") == pytest.approx(np.median(self.VALS))
        assert _agg(self.VALS, "unknown-how") == pytest.approx(np.median(self.VALS))

    def test_mean(self):
        assert _agg(self.VALS, "mean") == pytest.approx(np.mean(self.VALS))

    def test_trimmed_removes_tails(self):
        # trim 10% 后去掉 1.0 与 100.0 → mean(2..9) = 5.5
        assert _agg(self.VALS, "trimmed", trim_pct=10.0) == pytest.approx(5.5)

    def test_winsor_clips_tails(self):
        # 缩尾保留样本量：均值介于 trimmed 与原始 mean 之间
        w = _agg(self.VALS, "winsor", trim_pct=10.0)
        assert _agg(self.VALS, "trimmed", trim_pct=10.0) < w < np.mean(self.VALS)

    def test_avg3_is_mean_of_three(self):
        expected = np.mean([
            _agg(self.VALS, "median"),
            _agg(self.VALS, "trimmed", trim_pct=10.0),
            _agg(self.VALS, "mean"),
        ])
        assert _agg(self.VALS, "avg3", trim_pct=10.0) == pytest.approx(expected)

    def test_small_batch_falls_back_to_median(self):
        # <5 个谱时 trimmed/winsor 退化为 median（避免小样本截断失真）
        small = [1.0, 2.0, 100.0]
        assert _agg(small, "trimmed") == pytest.approx(2.0)
        assert _agg(small, "winsor") == pytest.approx(2.0)


class TestCoalRmse:
    # (pred, true, fold_train_mean) 批次三元组
    BATCHES = [(110.0, 100.0, 105.0), (95.0, 100.0, 105.0)]

    def test_no_shrink_is_pure_model_rmse(self):
        # w=1: errors = [10, -5] → sqrt((100+25)/2)
        assert coal_rmse(self.BATCHES, w=1.0) == pytest.approx(np.sqrt(62.5))

    def test_full_shrink_is_anchor_rmse(self):
        # w=0: 预测全部收缩到折内均值 105 → errors = [5, 5]
        assert coal_rmse(self.BATCHES, w=0.0) == pytest.approx(5.0)

    def test_partial_shrink_hand_computed(self):
        # w=0.8: pred = 0.8*p + 0.2*fm → [109, 97] → errors = [9, -3]
        assert coal_rmse(self.BATCHES, w=0.8) == pytest.approx(np.sqrt(45.0))

    def test_pooled_not_averaged(self):
        # pooled RMSE ≠ 各批次 |误差| 均值（口径检查）
        pooled = coal_rmse(self.BATCHES, w=1.0)
        mean_abs = np.mean([abs(110.0 - 100.0), abs(95.0 - 100.0)])
        assert pooled != pytest.approx(mean_abs)
