"""
Part 1 入口 — 预训练模型质量评测

逐个训练 AE/MAE/Contrastive，从三个维度评测:
  1. 重构误差 (AE/MAE)
  2. 隐变量-Y 相关性
  3. 线性探针 CV-RMSE (对比 PCA 基线)

用法:
  python eval_pretrain.py                         # 全量运行
  python eval_pretrain.py --methods ae,contrastive # 仅指定方法
  python eval_pretrain.py --latent-dims 16,32      # 仅指定维度
  python eval_pretrain.py --quick                  # 快速模式 (epochs=50)
"""

import argparse
import sys
import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR
from src.data import load_labels, load_coal_spectra
from src.pretrain import train_pretrain, PRETRAIN_METHODS, extract_latent_features
from src.pretrain_eval import (
    full_eval_report, print_report,
    eval_pca_baseline, print_comparison_table
)

PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output', 'pretrained'
)


def save_encoder(encoder, method, latent_dim, extra_info=None):
    """保存 encoder 供 Part 2 使用"""
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    path = os.path.join(PRETRAINED_DIR, f'{method}_latent{latent_dim}.pt')
    torch.save({
        'state_dict': encoder.state_dict(),
        'method': method,
        'latent_dim': latent_dim,
        'input_dim': encoder.encoder.net[0].in_features,
    }, path)
    print(f"  [保存] {path}")
    return path


def save_report(report):
    """保存评测报告 JSON"""
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    safe_name = f"{report['method']}_latent{report['latent_dim']}"
    # 转换不可序列化类型
    clean = {}
    for k, v in report.items():
        if isinstance(v, dict):
            clean[k] = {str(kk): float(vv) if isinstance(vv, (np.floating,)) else vv
                         for kk, vv in v.items()}
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        elif isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, list):
            clean[k] = [float(x) if isinstance(x, (np.floating,)) else x for x in v]
        else:
            clean[k] = v
    path = os.path.join(PRETRAINED_DIR, f'{safe_name}_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"  [保存] {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Part 1: 预训练模型质量评测')
    parser.add_argument('--methods', type=str, default=None,
                        help=f"逗号分隔: {','.join(PRETRAIN_METHODS)}")
    parser.add_argument('--latent-dims', type=str, default='8,16,32',
                        help='逗号分隔隐变量维度，默认 8,16,32')
    parser.add_argument('--epochs', type=int, default=200,
                        help='预训练最大轮数 (默认 200)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: epochs=50')
    args = parser.parse_args()

    epochs = 50 if args.quick else args.epochs
    methods = args.methods.split(',') if args.methods else PRETRAIN_METHODS
    latent_dims = [int(d) for d in args.latent_dims.split(',')]

    print(f"{'='*60}")
    print(f"Part 1: 预训练模型质量评测")
    print(f"  方法: {methods}")
    print(f"  隐变量维度: {latent_dims}")
    print(f"  最大轮数: {epochs}")
    print(f"{'='*60}\n")

    # ── 加载数据 ──
    print("加载数据...")
    label_map, aux_map = load_labels()
    train_data_dicts = []
    test_data_dicts = []
    for ct in COAL_TYPES:
        td = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        train_data_dicts.append(td)
        # 测试集不需要标签，用于推理一致性检查
        ted = load_coal_spectra(TEST_DIR, ct, None, None)
        test_data_dicts.append(ted)
        print(f"  {ct}: 训练 {len(td['spectra']) if td else 0} 条 / "
              f"测试 {len(ted['spectra']) if ted else 0} 条")
    print()

    # ── PCA 基线 ──
    print("\n计算 PCA 基线线性探针...")
    train_dict = {ct: d for ct, d in zip(COAL_TYPES, train_data_dicts) if d is not None}
    pca_baseline = eval_pca_baseline(train_dict)
    print(f"  PCA 线性探针 CV-RMSE: {pca_baseline['linear_probe_rmse']:.2f}")
    pca_baseline_rmse = pca_baseline['linear_probe_rmse']

    # ── 逐个训练 + 评测 ──
    all_reports = []

    for method in methods:
        for latent_dim in latent_dims:
            print(f"\n{'─'*60}")
            print(f"训练 {method.upper()} latent_dim={latent_dim}")
            print(f"{'─'*60}")

            try:
                encoder, history = train_pretrain(
                    train_data_dicts, method, latent_dim,
                    epochs=epochs,
                )

                # 保存 encoder
                save_encoder(encoder, method, latent_dim)

                # 评测
                report = full_eval_report(
                    encoder, train_data_dicts, test_data_dicts,
                    method, latent_dim
                )

                print_report(report)
                save_report(report)
                all_reports.append(report)

            except Exception as e:
                print(f"  [错误] {method}-{latent_dim}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── 对比总表 ──
    if all_reports:
        print_comparison_table(all_reports, pca_baseline_rmse)

    # ── 总结写入 ──
    print(f"\n评测完成！结果保存在 {PRETRAINED_DIR}/")
    print("后续 Part 2 将使用这些预训练编码器 + 各种预测模型进行组合评测。")


if __name__ == '__main__':
    import numpy as np
    main()
