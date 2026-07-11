"""
Experiment Tracker — 跨训练轮次记录实验元数据到 CSV

每次训练完成后追加一行，包含:
  - timestamp  : 训练结束时间戳
  - cv_rmse    : 全局交叉验证 RMSE
  - test_score : 留空，供用户后续填入测试集得分
  - treatment  : 留空，供用户后续填入本次尝试的处理方案描述
"""

import os
import csv
from datetime import datetime

TRACKER_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "experiment_log.csv",
)

HEADERS = ["timestamp", "cv_rmse", "test_score", "treatment"]


def _ensure_file():
    """若 CSV 文件不存在，创建并写入表头。"""
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    if not os.path.isfile(TRACKER_FILE):
        with open(TRACKER_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
        print(f"  [Tracker] 创建实验日志: {TRACKER_FILE}")


def log_experiment(cv_rmse: float) -> None:
    """
    追加一条实验记录。

    Parameters
    ----------
    cv_rmse : float
        本轮训练的全局交叉验证 RMSE。
    """
    _ensure_file()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(TRACKER_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([now, f"{cv_rmse:.2f}", "", ""])

    print(f"  [Tracker] 已记录 ↓")
    print(f"    timestamp  : {now}")
    print(f"    cv_rmse    : {cv_rmse:.2f}")
    print(f"    test_score : (待填写)")
    print(f"    treatment  : (待填写)")
