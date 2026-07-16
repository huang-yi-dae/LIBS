"""
Part 2 入口 — 预训练模型 + 预测模型组合评测

枚举所有 (预训练方法, 预测器) 组合，每组跑完整 GroupKFold CV，
输出对比表，比较 CV-RMSE。

预训练方法 (来自 Part 1):
  AE-8/16/32, MAE-8/16/32, Contrastive-8/16/32

预测器:
  RidgeCV, XGBoost, RandomForest, GBR, MLP

用法:
  python eval_combined.py                               # 全量组合
  python eval_combined.py --predictors ridge,xgboost     # 仅指定预测器
  python eval_combined.py --pretrained ae,contrastive    # 仅指定预训练方法
  python eval_combined.py --quick                        # 快速模式 (MLP max_iter=100)
"""

import argparse
import sys
import os
import json
import pickle
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR, SMALL_BATCH_THRESHOLD, ALPHAS
from src.data import load_labels, load_coal_spectra
from src.features import compute_features, build_feature_matrix
from src.model import get_cv_splits, aggregate_to_batch
from src.pretrain import (extract_latent_features, Autoencoder,
                          MaskedAutoencoder, ContrastiveEncoder)
from src.predictors import (
    PREDICTOR_NAMES, train_predictor_two_stage, predict_predictor_two_stage,
    train_predictor_direct, predict_predictor_direct
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output', 'pretrained'
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output'
)


# ── 扫描已训练的预训练模型 ────────────────────────────────────────────────────

def scan_pretrained():
    """扫描 output/pretrained/ 中保存的 encoder，返回 [(method, latent_dim, path)]"""
    results = []
    if not os.path.isdir(PRETRAINED_DIR):
        return results
    for fname in sorted(os.listdir(PRETRAINED_DIR)):
        if fname.endswith('.pt') and not fname.startswith('_'):
            # 格式: ae_latent8.pt
            parts = fname.replace('.pt', '').split('_')
            if len(parts) >= 2 and parts[-1].startswith('latent'):
                method = '_'.join(parts[:-1])
                latent_dim = int(parts[-1].replace('latent', ''))
                results.append((method, latent_dim, os.path.join(PRETRAINED_DIR, fname)))
    return results


def load_encoder(method, latent_dim, path):
    """加载预训练编码器"""
    checkpoint = torch.load(path, map_location='cpu')

    # 从 checkpoint 恢复 input_dim
    input_dim = checkpoint.get('input_dim', 7305)

    if method == 'ae':
        model = Autoencoder(input_dim, latent_dim)
    elif method == 'mae':
        model = MaskedAutoencoder(input_dim, latent_dim)
    elif method == 'contrastive':
        model = ContrastiveEncoder(input_dim, latent_dim)
    else:
        raise ValueError(f"未知方法: {method}")

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# ── 带预训练特征的特征提取 ────────────────────────────────────────────────────

def build_pretrain_feature_matrix(data, encoder):
    """
    用预训练编码器替换 PCA，构建特征矩阵。

    返回: X (n_spectra, latent_dim + hand_dim), 已标准化
    """
    # compute_features 填充手动特征
    inorm_mat = compute_features(data).astype(np.float32)

    # 预训练隐变量
    with torch.no_grad():
        latent = encoder.encode(torch.from_numpy(inorm_mat)).numpy()

    # 手工特征
    hand_feats = np.hstack([
        data['stats'], data['labs'], data['lrel'], data['rats']
    ])

    X = np.hstack([latent, hand_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, inorm_mat


# ── 单组合评测 ────────────────────────────────────────────────────────────────

def eval_one_combination(data_dicts, encoder, predictor_name,
                         predictor_params=None, mode='two-stage'):
    """
    评测一个 (预训练编码器, 预测器) 组合。

    参数:
        data_dicts: list[dict], 各煤种 data dict
        encoder: 预训练编码器
        predictor_name: 预测器名称
        predictor_params: 预测器参数
        mode: 'two-stage' 或 'direct'

    返回: dict {煤种: RMSE}, global_rmse
    """
    from src.model import get_cv_splits, aggregate_to_batch

    per_coal_results = {}
    all_batch_entries = []  # (rmse, n_test_lines)

    for ct_idx, ct in enumerate(COAL_TYPES):
        data = data_dicts[ct_idx]
        if data is None:
            continue

        # 构建特征: [预训练隐变量, 手工特征]
        X, _ = build_pretrain_feature_matrix(data, encoder)
        y = data['targets']
        aux_targets = data['aux']
        groups = data['groups']
        n_batches = data['n_batches']

        # CV 分割
        splits = get_cv_splits(groups, n_batches)

        # OOF 预测
        oof_preds = np.empty(len(y))

        for train_idx, val_idx in splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            if mode == 'two-stage' and aux_targets is not None:
                aux_tr = aux_targets[train_idx]
                model_dict = train_predictor_two_stage(
                    predictor_name, X_tr, y_tr, aux_tr, predictor_params
                )
                oof_preds[val_idx] = predict_predictor_two_stage(model_dict, X_val)
            else:
                # direct mode — X → Q
                # 但为了公平比较，two-stage 预测器也支持 direct fallback
                model_dict = train_predictor_direct(
                    predictor_name, X_tr, y_tr, predictor_params
                )
                oof_preds[val_idx] = predict_predictor_direct(model_dict, X_val)

        # 批次聚合
        unique_groups = np.unique(groups)
        batch_rmses = []
        batch_sizes = []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            batch_pred = float(np.median(oof_preds[idx]))
            batch_true = float(np.median(y[idx]))
            batch_sizes.append(len(idx))
            batch_rmses.append((batch_pred - batch_true) ** 2)

        coal_rmse = float(np.sqrt(np.mean(batch_rmses))) if batch_rmses else 999.0
        per_coal_results[ct] = coal_rmse
        all_batch_entries.append((coal_rmse, n_batches))

    # 全局 pooled RMSE（按测试批次数加权）
    if not all_batch_entries:
        return per_coal_results, 999.0

    rmses = np.array([r for r, _ in all_batch_entries])
    weights = np.array([n for _, n in all_batch_entries], dtype=np.float32)
    global_rmse = float(np.sqrt(np.average(rmses ** 2, weights=weights)))

    return per_coal_results, global_rmse


# ── PCA 基线 (使用 build_feature_matrix + 指定的预测器) ────────────────────────

def eval_pca_with_predictor(data_dicts, predictor_name, predictor_params=None, mode='two-stage'):
    """
    用 PCA 特征替换预训练特征，其余相同的评测流程。
    作为对照基线。
    """
    from src.model import get_cv_splits, aggregate_to_batch

    per_coal_results = {}
    all_batch_entries = []

    for ct_idx, ct in enumerate(COAL_TYPES):
        data = data_dicts[ct_idx]
        if data is None:
            continue

        # PCA 特征
        X, _, _, _ = build_feature_matrix(
            data, n_batches=data['n_batches'], fit=True, perturb_cfg=None
        )
        y = data['targets']
        aux_targets = data['aux']
        groups = data['groups']
        n_batches = data['n_batches']

        splits = get_cv_splits(groups, n_batches)
        oof_preds = np.empty(len(y))

        for train_idx, val_idx in splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            if mode == 'two-stage' and aux_targets is not None:
                aux_tr = aux_targets[train_idx]
                model_dict = train_predictor_two_stage(
                    predictor_name, X_tr, y_tr, aux_tr, predictor_params
                )
                oof_preds[val_idx] = predict_predictor_two_stage(model_dict, X_val)
            else:
                model_dict = train_predictor_direct(
                    predictor_name, X_tr, y_tr, predictor_params
                )
                oof_preds[val_idx] = predict_predictor_direct(model_dict, X_val)

        unique_groups = np.unique(groups)
        batch_rmses = []
        batch_sizes = []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            batch_pred = float(np.median(oof_preds[idx]))
            batch_true = float(np.median(y[idx]))
            batch_sizes.append(len(idx))
            batch_rmses.append((batch_pred - batch_true) ** 2)

        coal_rmse = float(np.sqrt(np.mean(batch_rmses))) if batch_rmses else 999.0
        per_coal_results[ct] = coal_rmse
        all_batch_entries.append((coal_rmse, n_batches))

    if not all_batch_entries:
        return per_coal_results, 999.0
    rmses = np.array([r for r, _ in all_batch_entries])
    weights = np.array([n for _, n in all_batch_entries], dtype=np.float32)
    global_rmse = float(np.sqrt(np.average(rmses ** 2, weights=weights)))
    return per_coal_results, global_rmse


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Part 2: 组合评测')
    parser.add_argument('--predictors', type=str, default=None,
                        help=f"逗号分隔: {','.join(PREDICTOR_NAMES.keys())}")
    parser.add_argument('--pretrained', type=str, default=None,
                        help='逗号分隔预训练方法: ae,mae,contrastive')
    parser.add_argument('--mode', type=str, default='two-stage',
                        choices=['two-stage', 'direct'],
                        help='预测模式: two-stage (使用辅助指标) 或 direct (直接预测)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: MLP max_iter=100')
    args = parser.parse_args()

    predictor_names = args.predictors.split(',') if args.predictors else list(PREDICTOR_NAMES.keys())
    pretrained_filters = args.pretrained.split(',') if args.pretrained else None

    print(f"{'='*60}")
    print(f"Part 2: 预训练模型 + 预测模型 组合评测")
    print(f"  预测器: {predictor_names}")
    print(f"  模式: {args.mode}")
    print(f"{'='*60}\n")

    # ── 加载数据 ──
    print("加载数据...")
    label_map, aux_map = load_labels()
    data_dicts = []
    for ct in COAL_TYPES:
        td = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        data_dicts.append(td)
        print(f"  {ct}: {len(td['spectra']) if td else 0} 条, {td['n_batches'] if td else 0} 批次")

    # ── 扫描预训练模型 ──
    all_pretrained = scan_pretrained()
    if not all_pretrained:
        print("\n[错误] 未找到预训练模型。请先运行 eval_pretrain.py")
        sys.exit(1)

    if pretrained_filters:
        all_pretrained = [
            (m, d, p) for m, d, p in all_pretrained
            if m in pretrained_filters
        ]

    print(f"\n找到 {len(all_pretrained)} 个预训练模型:")
    for method, latent_dim, path in all_pretrained:
        print(f"  {method.upper()}-{latent_dim}  → {os.path.basename(path)}")

    # ── 创建结果矩阵 ──
    # 行: 预训练模型 列: 预测器
    # 格式: {(method, latent_dim): {predictor: cv_rmse}}
    results = {}

    total = len(all_pretrained) * len(predictor_names)
    done = 0

    for method, latent_dim, path in all_pretrained:
        key = f"{method.upper()}-{latent_dim}"
        print(f"\n{'─'*60}")
        print(f"加载编码器: {key}")
        print(f"{'─'*60}")

        encoder = load_encoder(method, latent_dim, path)
        results[(method, latent_dim)] = {}

        for pred_name in predictor_names:
            done += 1
            display_name = PREDICTOR_NAMES.get(pred_name, pred_name)

            # 快速模式下 MLP 减少 iter
            params = None
            if args.quick and pred_name == 'mlp':
                params = {'max_iter': 100}

            print(f"  [{done}/{total}] 评测 {key} + {display_name} ...", end=' ', flush=True)

            try:
                per_coal, global_rmse = eval_one_combination(
                    data_dicts, encoder, pred_name,
                    predictor_params=params, mode=args.mode
                )
                results[(method, latent_dim)][pred_name] = global_rmse
                print(f"CV-RMSE = {global_rmse:.2f}")
            except Exception as e:
                print(f"[错误] {e}")
                results[(method, latent_dim)][pred_name] = 999.0
                import traceback
                traceback.print_exc()

    # ── PCA 基线 ──
    print(f"\n{'─'*60}")
    print("计算 PCA 基线...")
    print(f"{'─'*60}")
    pca_results = {}
    for pred_name in predictor_names:
        display_name = PREDICTOR_NAMES.get(pred_name, pred_name)
        params = None
        if args.quick and pred_name == 'mlp':
            params = {'max_iter': 100}
        print(f"  基线: PCA + {display_name} ...", end=' ', flush=True)
        try:
            _, global_rmse = eval_pca_with_predictor(
                data_dicts, pred_name, params, mode=args.mode
            )
            pca_results[pred_name] = global_rmse
            print(f"CV-RMSE = {global_rmse:.2f}")
        except Exception as e:
            print(f"[错误] {e}")
            pca_results[pred_name] = 999.0

    # ── 输出对比表 ──
    print(f"\n\n{'='*100}")
    print(f"  Part 2 — 组合评测对比表 (模式: {args.mode})")
    print(f"{'='*100}")

    # 列: 预测器
    col_w = max(16, max(len(PREDICTOR_NAMES.get(p, p)) for p in predictor_names) + 2)
    header = f"{'特征':<24}"
    for p in predictor_names:
        header += f" {PREDICTOR_NAMES.get(p, p):>{col_w}}"
    print(header)
    print("-" * 100)

    # 行: 每个预训练模型
    for method, latent_dim in sorted(results.keys()):
        key = f"{method.upper()}-{latent_dim}"
        row = f"{key:<24}"
        for p in predictor_names:
            val = results[(method, latent_dim)].get(p, 999.0)
            best_in_col = min(v.get(p, 999.0) for v in results.values())
            s = f" {val:>{col_w}.1f}"
            if val == best_in_col and val < min(pca_results.get(p, 999.0), 900):
                s = f" ★{val:>{col_w-2}.1f}"
            row += s
        print(row)

    # PCA 基线行
    pca_row = f"{'PCA (baseline)':<24}"
    for p in predictor_names:
        val = pca_results.get(p, 999.0)
        best_in_col = min(
            [v.get(p, 999.0) for v in results.values()] + [val]
        )
        best_in_row = min(
            [pca_results.get(pp, 999.0) for pp in predictor_names]
        )
        s = f" {val:>{col_w}.1f}"
        if val == best_in_col:
            s = f" ★{val:>{col_w-2}.1f}"
        pca_row += s
    print("-" * 100)
    print(pca_row)
    print("=" * 100)
    print("  ★ = 列最优 (低于 900 且优于 PCA 基线)")

    # ── 保存结果 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, 'combined_eval_results.json')
    save_data = {
        'mode': args.mode,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {
            f'{m.upper()}-{d}': {PREDICTOR_NAMES.get(p, p): v for p, v in preds.items()}
            for (m, d), preds in results.items()
        },
        'pca_baseline': {PREDICTOR_NAMES.get(p, p): v for p, v in pca_results.items()},
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()
