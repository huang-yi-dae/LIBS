"""
预训练模型评测模块

评测维度:
  1. 重构误差 (AE/MAE) — MSE, R²
  2. 隐变量-Y 相关性 — 各维度 Pearson r, max/mean |r|
  3. 线性探针 — RidgeCV 在冻结隐变量上预测发热量 → CV-RMSE (与 PCA 基线对比)
  4. t-SNE 可视化 (打印报告)
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler

from config import ALPHAS, SMALL_BATCH_THRESHOLD, COAL_TYPES, N_PCA_MAX, RANDOM_STATE
from src.features import compute_features, build_feature_matrix
from src.data import load_coal_spectra
from src.pretrain import extract_latent_features


# ── 工具: CV 分割 ─────────────────────────────────────────────────────────────

def _cv_splits(groups, n_batches):
    """同 model.py: LOOCV 或 GroupKFold(5)"""
    dummy = np.zeros(len(groups))
    if n_batches <= SMALL_BATCH_THRESHOLD:
        return list(LeaveOneGroupOut().split(dummy, dummy, groups))
    k = min(5, n_batches)
    return list(GroupKFold(n_splits=k).split(dummy, dummy, groups))


# ══════════════════════════════════════════════════════════════════════════════
# 1. 重构误差评测
# ══════════════════════════════════════════════════════════════════════════════

def eval_reconstruction(encoder, data_dicts, method_name):
    """
    计算重构 MSE 和 R²。

    仅对 AE/MAE 有效（需要有 decoder）。
    返回 dict。
    """
    import torch
    from src.pretrain import DEVICE

    all_orig, all_recon = [], []
    model = encoder.to(DEVICE)
    model.eval()

    for data in data_dicts:
        if data is None:
            continue
        inorm = compute_features(data).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(inorm).to(DEVICE))
            # MAE forward 返回 (recon, mask)，AE 返回 tensor
            if isinstance(out, tuple):
                out = out[0]
            recon = out.cpu().numpy()
        all_orig.append(inorm)
        all_recon.append(recon)

    X_orig = np.vstack(all_orig)
    X_recon = np.vstack(all_recon)

    mse = float(mean_squared_error(X_orig, X_recon))
    # R²: 每样本的 R² 取平均
    ss_res = ((X_orig - X_recon) ** 2).sum(axis=1)
    ss_tot = ((X_orig - X_orig.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2_per_sample = 1 - ss_res / (ss_tot + 1e-12)
    r2 = float(np.mean(r2_per_sample))

    return {
        'method': method_name,
        'recon_mse': mse,
        'recon_r2': r2,
        'recon_mse_per_point': float(((X_orig - X_recon) ** 2).mean(axis=0).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 隐变量-Y 相关性
# ══════════════════════════════════════════════════════════════════════════════

def eval_latent_correlation(encoder, data_dicts, method_name):
    """
    计算隐变量各维度与发热量 (targets) 的 Pearson 相关系数。

    返回 dict，包含 max/mean/abs 等统计量。
    """
    all_latent, all_y = [], []
    for data in data_dicts:
        if data is None or data['targets'] is None:
            continue
        latent = extract_latent_features(encoder, data)
        all_latent.append(latent)
        all_y.append(data['targets'])

    Z = np.vstack(all_latent)
    y = np.concatenate(all_y)

    # 逐维度 Pearson r
    r_vals = np.array([pearsonr(Z[:, i], y)[0] for i in range(Z.shape[1])])

    return {
        'method': method_name,
        'latent_dim': Z.shape[1],
        'corr_max_abs': float(np.abs(r_vals).max()),
        'corr_mean_abs': float(np.abs(r_vals).mean()),
        'corr_max': float(r_vals.max()),
        'corr_min': float(r_vals.min()),
        'corr_best_dim': int(np.argmax(np.abs(r_vals))),
        'corr_all_dims': r_vals.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 线性探针: 冻结隐变量 → RidgeCV 预测发热量
# ══════════════════════════════════════════════════════════════════════════════

def eval_linear_probe(encoder, train_data_dicts, test_data_dicts, method_name):
    """
    用冻结的隐变量 + 手工特征，替代 PCA 特征，GroupKFold CV 评测。

    模拟与主流水线相同的评测方式:
      - 预训练特征 + 手工特征 → RidgeCV → 预测发热量
      - 批次数加权 pooled CV-RMSE

    返回 dict。
    """
    from src.model import aggregate_to_batch

    per_coal_results = {}
    all_oof_batches = []  # (rmse, n_test) for global pooled

    for ct in COAL_TYPES:
        train_data = train_data_dicts.get(ct)
        test_data  = test_data_dicts.get(ct) if test_data_dicts else None
        if train_data is None:
            continue

        # 提取预训练隐变量
        latent = extract_latent_features(encoder, train_data)

        # 手工特征（已在 extract_latent_features 中由 compute_features 填充）
        hand_feats = np.hstack([
            train_data['stats'], train_data['labs'],
            train_data['lrel'], train_data['rats']
        ])
        X = np.hstack([latent, hand_feats])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = train_data['targets']
        groups = train_data['groups']
        n_batches = train_data['n_batches']

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # CV 分割
        splits = _cv_splits(groups, n_batches)

        # 内置 RidgeCV 交叉验证
        model = RidgeCV(alphas=ALPHAS, scoring='neg_mean_squared_error')
        model.fit(X_scaled, y)

        # OOF 预测
        oof_preds = np.empty(len(y))
        for train_idx, val_idx in splits:
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr = y[train_idx]
            m = RidgeCV(alphas=ALPHAS).fit(X_tr, y_tr)
            oof_preds[val_idx] = m.predict(X_val)

        # 批次聚合 + 计算批次 RMSE
        # 需知道 批次名 → group index 的映射
        unique_groups = np.unique(groups)
        batch_rmses = []
        batch_sizes = []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            batch_pred = np.median(oof_preds[idx])
            batch_true = np.median(y[idx])
            batch_sizes.append(len(idx))
            batch_rmses.append((batch_pred - batch_true) ** 2)

        coal_rmse = float(np.sqrt(np.mean(batch_rmses)))
        n_test = len(unique_groups)
        per_coal_results[ct] = coal_rmse
        all_oof_batches.append((coal_rmse, n_test))

    # 全局 pooled RMSE（按测试批次加权，与线上对齐）
    rmses = np.array([r for r, _ in all_oof_batches])
    weights = np.array([n for _, n in all_oof_batches], dtype=np.float32)
    global_rmse = float(np.sqrt(np.average(rmses ** 2, weights=weights)))

    return {
        'method': method_name,
        'linear_probe_rmse': global_rmse,
        'per_coal_rmse': per_coal_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PCA 基线探针
# ══════════════════════════════════════════════════════════════════════════════

def eval_pca_baseline(train_data_dicts, test_data_dicts=None):
    """
    用 PCA 特征替换编码器，其余完全相同 → 获得基线线性探针 RMSE。
    """
    per_coal_results = {}
    all_oof_batches = []

    for ct in COAL_TYPES:
        train_data = train_data_dicts.get(ct)
        if train_data is None:
            continue

        # 用标准 build_feature_matrix 获得 PCA + 手工特征
        X, _, _, _ = build_feature_matrix(
            train_data, n_batches=train_data['n_batches'],
            fit=True, perturb_cfg=None
        )
        y = train_data['targets']
        groups = train_data['groups']
        n_batches = train_data['n_batches']

        splits = _cv_splits(groups, n_batches)

        oof_preds = np.empty(len(y))
        for train_idx, val_idx in splits:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            m = RidgeCV(alphas=ALPHAS).fit(X_tr, y_tr)
            oof_preds[val_idx] = m.predict(X_val)

        unique_groups = np.unique(groups)
        batch_rmses = []
        batch_sizes = []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            batch_pred = np.median(oof_preds[idx])
            batch_true = np.median(y[idx])
            batch_sizes.append(len(idx))
            batch_rmses.append((batch_pred - batch_true) ** 2)

        coal_rmse = float(np.sqrt(np.mean(batch_rmses)))
        n_test = len(unique_groups)
        per_coal_results[ct] = coal_rmse
        all_oof_batches.append((coal_rmse, n_test))

    rmses = np.array([r for r, _ in all_oof_batches])
    weights = np.array([n for _, n in all_oof_batches], dtype=np.float32)
    global_rmse = float(np.sqrt(np.average(rmses ** 2, weights=weights)))

    return {
        'method': 'PCA (baseline)',
        'linear_probe_rmse': global_rmse,
        'per_coal_rmse': per_coal_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 完整评测报告
# ══════════════════════════════════════════════════════════════════════════════

def full_eval_report(encoder, data_dicts_train, data_dicts_test, method_name,
                     latent_dim):
    """
    运行所有评测（除 PCA 基线外），返回完整报告 dict。
    """
    report = {'method': method_name, 'latent_dim': latent_dim}

    # 重构评测
    if method_name in ('ae', 'mae'):
        try:
            report.update(eval_reconstruction(encoder, data_dicts_train, method_name))
        except Exception as e:
            report['recon_error'] = str(e)

    # 隐变量-Y 相关性
    try:
        report.update(eval_latent_correlation(encoder, data_dicts_train, method_name))
    except Exception as e:
        report['corr_error'] = str(e)

    # 线性探针
    try:
        train_dict = {ct: d for ct, d in zip(COAL_TYPES, data_dicts_train) if d is not None}
        test_dict = {ct: d for ct, d in zip(COAL_TYPES, data_dicts_test) if d is not None} if data_dicts_test else None
        report.update(eval_linear_probe(encoder, train_dict, test_dict, method_name))
    except Exception as e:
        report['probe_error'] = str(e)

    return report


def print_report(report):
    """打印单方法评测报告"""
    m = report['method'].upper()
    d = report['latent_dim']
    print(f"\n{'='*60}")
    print(f"  {m} (latent_dim={d}) 评测报告")
    print(f"{'='*60}")

    if 'recon_mse' in report:
        print(f"  重构 MSE : {report['recon_mse']:.6f}")
        print(f"  重构 R2  : {report['recon_r2']:.4f}")

    if 'corr_max_abs' in report:
        print(f"  隐变量-Y 相关性:")
        print(f"    max |r| = {report['corr_max_abs']:.4f}  (dim {report.get('corr_best_dim', '?')})")
        print(f"    mean|r| = {report['corr_mean_abs']:.4f}")
        print(f"    r range = [{report.get('corr_min', '?'):.4f}, {report.get('corr_max', '?'):.4f}]")

    if 'linear_probe_rmse' in report:
        print(f"  线性探针 (RidgeCV) CV-RMSE: {report['linear_probe_rmse']:.2f}")
        if 'per_coal_rmse' in report:
            for ct, rmse in report['per_coal_rmse'].items():
                print(f"    {ct}: {rmse:.2f}")

    if 'recon_error' in report:
        print(f"  [警告] 重构评测失败: {report['recon_error']}")
    if 'corr_error' in report:
        print(f"  [警告] 相关性评测失败: {report['corr_error']}")
    if 'probe_error' in report:
        print(f"  [警告] 线性探针评测失败: {report['probe_error']}")


def print_comparison_table(reports, pca_baseline_rmse=None):
    """
    打印多方法对比表（含 PCA 基线）。
    """
    print(f"\n{'='*80}")
    print(f"  Part 1 — 预训练模型质量对比总表")
    print(f"{'='*80}")

    header = f"{'方法':<24} {'隐变量':>6} {'重构MSE':>12} {'重构R²':>8} {'max|r|':>8} {'mean|r|':>8} {'探针RMSE':>10}"
    print(header)
    print("-" * 80)

    for r in reports:
        m = f"{r['method'].upper()}-{r['latent_dim']}"
        ld = str(r['latent_dim'])
        rm = f"{r.get('recon_mse', -1):.4f}" if 'recon_mse' in r else "N/A"
        r2 = f"{r.get('recon_r2', -1):.3f}" if 'recon_r2' in r else "N/A"
        mc = f"{r.get('corr_max_abs', -1):.3f}" if 'corr_max_abs' in r else "N/A"
        mac = f"{r.get('corr_mean_abs', -1):.3f}" if 'corr_mean_abs' in r else "N/A"
        lp = f"{r.get('linear_probe_rmse', -1):.1f}" if 'linear_probe_rmse' in r else "N/A"
        print(f"{m:<24} {ld:>6} {rm:>12} {r2:>8} {mc:>8} {mac:>8} {lp:>10}")

    if pca_baseline_rmse is not None:
        print("-" * 80)
        print(f"{'PCA 基线 (N_PCA_MAX=30)':<24} {'—':>6} {'—':>12} {'—':>8} {'—':>8} {'—':>8} {pca_baseline_rmse:>10.1f}")

    print("=" * 80)
