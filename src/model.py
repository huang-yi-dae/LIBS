"""
两阶段回归模型

Stage 1 — 光谱特征 → 辅助指标 (全水分/灰分/氢/硫)
  - 辅助指标与发热量相关性强，但比发热量本身更稳定
  - 用 OOF(Out-of-Fold) 预测避免数据泄露进 Stage2

Stage 2 — 光谱特征 + Stage1预测辅助指标 → 发热量
  - 信息融合: 数值模型 + 物理知识

均值收缩 (Mean Shrinkage):
  - 批次极少(≤10)的煤种，模型容易过拟合到训练批次的特殊性
  - 将预测值向煤种均值收缩: pred = w * model_pred + (1-w) * coal_mean
  - w 通过 OOF 验证集上网格搜索确定
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

from config import ALPHAS, AUX_COLS, SMALL_BATCH_THRESHOLD
from src.features import build_feature_matrix


# ── CV 分割策略 ───────────────────────────────────────────────────────────────

def get_cv_splits(groups, n_batches):
    """
    批次数 ≤ 10 时用 LOOCV（留一批次），否则用 GroupKFold(5折)。
    关键: 以批次(group)为单位分割，而非以单条光谱为单位——
    同一批次的光谱不能同时出现在训练集和验证集中。
    """
    dummy = np.zeros(len(groups))
    if n_batches <= SMALL_BATCH_THRESHOLD:
        return list(LeaveOneGroupOut().split(dummy, dummy, groups))
    else:
        k = min(5, n_batches)
        return list(GroupKFold(n_splits=k).split(dummy, dummy, groups))


# ── 批次聚合 ──────────────────────────────────────────────────────────────────

def aggregate_to_batch(preds, names):
    """
    同批次多条光谱的预测值 → 取中位数作为批次预测。
    中位数比均值对异常光谱更鲁棒。
    """
    grouped = {}
    for p, n in zip(preds, names):
        grouped.setdefault(n, []).append(p)
    return {k: float(np.median(v)) for k, v in grouped.items()}


# ── 均值收缩搜索 ──────────────────────────────────────────────────────────────

def find_best_shrinkage(oof_preds, oof_true, coal_mean):
    """
    在 [0, 1] 上网格搜索混合权重 w，最小化 OOF-RMSE。
    w=1.0 代表纯模型，w=0.0 代表纯均值。
    """
    best_w, best_rmse = 1.0, float('inf')
    for w in np.linspace(0.0, 1.0, 21):
        blended = w * np.array(oof_preds) + (1 - w) * coal_mean
        rmse    = float(np.sqrt(np.mean((blended - np.array(oof_true)) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return best_w, best_rmse


# ── 单煤种训练 ────────────────────────────────────────────────────────────────

def train_coal_model(coal_type, train_data):
    """
    训练某煤种的两阶段模型，返回预测所需的全部参数。

    输出 dict 包含:
        spec_scalers  : (scaler_spec, pca, scaler_hand)
        aux_models    : {辅助指标名: RidgeCV 或 None}
        scaler_s2     : Stage2 的特征标准化器
        final_model   : Stage2 最终 RidgeCV
        coal_mean     : 训练集发热量均值（收缩锚点）
        shrink_w      : 收缩权重 w（1.0 = 不收缩）
        cv_rmse       : 交叉验证 RMSE
    """
    y          = train_data['targets']
    aux        = train_data['aux']
    groups     = train_data['groups']
    n_batches  = train_data['n_batches']
    coal_mean  = float(y.mean())

    print(f"\n  [{coal_type}]  {n_batches}批次  {len(y)}条光谱  "
          f"Q={y.min():.0f}~{y.max():.0f}")

    # 光谱 → 特征矩阵（训练集 fit）
    X_spec, scaler_spec, pca, scaler_hand = build_feature_matrix(
        train_data, n_batches, fit=True)

    splits = get_cv_splits(groups, n_batches)

    # ── Stage 1: 光谱 → 辅助指标 (OOF) ────────────────────────────────────
    aux_models        = {}
    predicted_aux_oof = np.zeros_like(aux, dtype=np.float32)

    for col_idx, col_name in enumerate(AUX_COLS):
        y_aux = aux[:, col_idx]

        # 辅助指标有缺失时退化为用均值填充
        if np.isnan(y_aux).any():
            predicted_aux_oof[:, col_idx] = float(np.nanmean(y_aux))
            aux_models[col_name] = None
            continue

        m = RidgeCV(alphas=ALPHAS)
        oof = np.zeros(len(y_aux))
        for tr_idx, val_idx in splits:
            m.fit(X_spec[tr_idx], y_aux[tr_idx])
            oof[val_idx] = m.predict(X_spec[val_idx])
        predicted_aux_oof[:, col_idx] = oof

        m.fit(X_spec, y_aux)   # 全量重新拟合，存入 aux_models 供推理用
        aux_models[col_name] = m

    # ── Stage 2: [光谱特征 + 预测辅助指标] → 发热量 ──────────────────────
    X_s2      = np.hstack([X_spec, predicted_aux_oof])
    scaler_s2 = StandardScaler()
    X_s2      = scaler_s2.fit_transform(np.nan_to_num(X_s2))

    # OOF 批次预测（用于计算 CV-RMSE 和收缩权重）
    oof_batch_preds, oof_batch_true, batch_rmses = [], [], []

    for tr_idx, val_idx in splits:
        m2 = RidgeCV(alphas=ALPHAS)
        m2.fit(X_s2[tr_idx], y[tr_idx])
        val_pred   = m2.predict(X_s2[val_idx])
        val_groups = groups[val_idx]

        fold_se = []
        for bg in np.unique(val_groups):
            mask   = val_groups == bg
            true_q = float(y[val_idx][mask][0])
            pred_q = float(np.median(val_pred[mask]))
            oof_batch_preds.append(pred_q)
            oof_batch_true.append(true_q)
            fold_se.append((true_q - pred_q) ** 2)
        batch_rmses.append(float(np.sqrt(np.mean(fold_se))))

    cv_rmse_raw = float(np.mean(batch_rmses))

    # 小批次煤种: 搜索最优收缩权重
    best_w = 1.0
    if n_batches <= SMALL_BATCH_THRESHOLD:
        best_w, cv_rmse_shrunk = find_best_shrinkage(
            oof_batch_preds, oof_batch_true, coal_mean)
        cv_rmse = cv_rmse_shrunk
        print(f"    Stage2 CV-RMSE: raw={cv_rmse_raw:.2f}  "
              f"shrunk={cv_rmse_shrunk:.2f}  w={best_w:.2f}")
    else:
        cv_rmse = cv_rmse_raw
        print(f"    Stage2 CV-RMSE: {cv_rmse:.2f} ± {np.std(batch_rmses):.2f}")

    # 全量数据重新拟合最终模型
    final_model = RidgeCV(alphas=ALPHAS)
    final_model.fit(X_s2, y)
    print(f"    最优正则化 alpha: {final_model.alpha_:.1f}")

    return {
        'spec_scalers': (scaler_spec, pca, scaler_hand),
        'aux_models':   aux_models,
        'scaler_s2':    scaler_s2,
        'final_model':  final_model,
        'coal_mean':    coal_mean,
        'shrink_w':     best_w,
        'cv_rmse':      cv_rmse,
    }


# ── 单煤种推理 ────────────────────────────────────────────────────────────────

def predict_coal(coal_type, test_data, model_dict):
    """
    对某煤种的测试批次做推理，返回 {批次名: 预测发热量} 字典。
    推理流程与训练流程一一对应:
      光谱 → [Stage1] → 预测辅助指标 → [Stage2] → 发热量 → 批次聚合
    """
    scaler_spec, pca, scaler_hand = model_dict['spec_scalers']
    aux_models  = model_dict['aux_models']
    scaler_s2   = model_dict['scaler_s2']
    final_model = model_dict['final_model']
    coal_mean   = model_dict['coal_mean']
    shrink_w    = model_dict['shrink_w']

    X_spec = build_feature_matrix(
        test_data, n_batches=None,
        scaler_spec=scaler_spec, pca=pca, scaler_hand=scaler_hand,
        fit=False)

    pred_aux = np.zeros((len(test_data['spectra']), len(AUX_COLS)), dtype=np.float32)
    for col_idx, col_name in enumerate(AUX_COLS):
        m = aux_models.get(col_name)
        if m is not None:
            pred_aux[:, col_idx] = m.predict(X_spec)

    X_s2       = scaler_s2.transform(np.nan_to_num(np.hstack([X_spec, pred_aux])))
    preds      = final_model.predict(X_s2)
    batch_pred = aggregate_to_batch(preds, test_data['names'])

    # 应用收缩（仅小批次煤种，且 w < 1.0 时）
    if shrink_w < 1.0:
        batch_pred = {
            k: shrink_w * v + (1 - shrink_w) * coal_mean
            for k, v in batch_pred.items()
        }

    return batch_pred
