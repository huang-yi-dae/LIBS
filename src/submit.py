"""
提交文件打包模块

生成符合赛题要求的 submit.zip，内部结构:
    submit/
    └── submit.csv     预测结果
"""

import os
import zipfile
import pandas as pd

from config import SUBMIT_TEMPLATE, OUTPUT_DIR


def pack_submission(all_preds: dict, cv_results: dict, global_cv_rmse: float):
    """
    生成 submit.csv 并打包为 submit.zip（仅含 submit/submit.csv）。

    参数:
        all_preds      : {批次名: 预测发热量}
        cv_results     : {煤种名: cv_rmse}（保留参数签名，不再生成 README）
        global_cv_rmse : 全局 CV-RMSE
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 读取提交模板（保证名称顺序正确）
    template = pd.read_csv(SUBMIT_TEMPLATE, encoding='utf-8')
    template['预测发热量_MJ_KG'] = template['名称'].map(all_preds)

    missing = template[template['预测发热量_MJ_KG'].isna()]['名称'].tolist()
    if missing:
        print(f"  警告: {len(missing)} 个批次缺少预测，用均值填充: {missing}")
        fallback = sum(all_preds.values()) / len(all_preds)
        template['预测发热量_MJ_KG'] = template['预测发热量_MJ_KG'].fillna(fallback)

    # ── 写 CSV
    csv_path = os.path.join(OUTPUT_DIR, "submit.csv")
    template.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ {csv_path}")
    print(template.to_string(index=False))

    # ── 打包 ZIP（仅 submit/submit.csv）
    zip_path = os.path.join(OUTPUT_DIR, "submit.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submit/submit.csv")

    print(f"  ✓ {zip_path}")
    print(f"     内含: submit/submit.csv")
    return zip_path
