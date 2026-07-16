"""
预测模型统一接口

提供多种预测器，统一 fit/predict 接口，支持两阶段或直接预测模式。

两阶段模式:
  Stage1: feature → predict aux indicators
  Stage2: feature + aux → predict Q (发热量)

直接模式:
  feature → predict Q

支持的预测器:
  - ridge:    RidgeCV (基线)
  - xgboost:  XGBRegressor
  - rf:       RandomForestRegressor
  - gbr:      GradientBoostingRegressor
  - mlp:      MLPRegressor
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import ALPHAS, AUX_COLS, RANDOM_STATE


# ── 预测器工厂 ────────────────────────────────────────────────────────────────

def _base_params():
    """各模型的基础参数字典，供后续调参使用"""
    return {
        'xgboost': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'verbosity': 0,
        },
        'rf': {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_leaf': 3,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        },
        'gbr': {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'min_samples_leaf': 3,
            'random_state': RANDOM_STATE,
        },
        'mlp': {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 32,
            'max_iter': 500,
            'early_stopping': True,
            'random_state': RANDOM_STATE,
        },
    }


PREDICTOR_NAMES = {
    'ridge':    'RidgeCV',
    'xgboost':  'XGBoost',
    'rf':       'RandomForest',
    'gbr':      'GBR',
    'mlp':      'MLP',
}


def create_predictor(name, params=None):
    """
    创建指定类型的预测器实例。

    参数:
        name: 'ridge' / 'xgboost' / 'rf' / 'gbr' / 'mlp'
        params: dict, 模型参数（覆盖默认值）

    返回: predictor 实例
    """
    base = _base_params().get(name, {})
    if params:
        base.update(params)

    if name == 'ridge':
        return RidgeCV(alphas=base.get('alphas', ALPHAS))
    elif name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(**base)
    elif name == 'rf':
        return RandomForestRegressor(**base)
    elif name == 'gbr':
        return GradientBoostingRegressor(**base)
    elif name == 'mlp':
        return MLPRegressor(**base)
    else:
        raise ValueError(f"未知预测器: {name}，可选: {list(PREDICTOR_NAMES.keys())}")


# ── 两阶段训练/推理 ────────────────────────────────────────────────────────────

def train_predictor_two_stage(name, X, y, aux_targets, params=None):
    """
    两阶段训练:
      Stage1: X → aux (每个辅助列独立训练)
      Stage2: [X, aux_pred] → Q

    返回: {'stage1_models': dict, 'stage2_model': model, 'scaler': StandardScaler}
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stage1: 每个辅助列单独建模
    stage1_models = {}
    for col_idx, col_name in enumerate(AUX_COLS):
        y_aux = aux_targets[:, col_idx]
        if np.isnan(y_aux).all():
            continue
        m = create_predictor(name, params)
        m.fit(X_scaled, y_aux)
        stage1_models[col_name] = m

    # Stage1 预测
    pred_aux = np.zeros((len(X), len(AUX_COLS)), dtype=np.float32)
    for col_idx, col_name in enumerate(AUX_COLS):
        m = stage1_models.get(col_name)
        if m is not None:
            pred_aux[:, col_idx] = m.predict(X_scaled)

    # Stage2: [X, aux_pred] → Q
    X_s2 = np.hstack([X_scaled, pred_aux])
    stage2_model = create_predictor(name, params)
    stage2_model.fit(X_s2, y)

    return {
        'stage1_models': stage1_models,
        'stage2_model': stage2_model,
        'scaler': scaler,
    }


def predict_predictor_two_stage(model_dict, X_test):
    """
    两阶段推理。
    """
    scaler = model_dict['scaler']
    stage1_models = model_dict['stage1_models']
    stage2_model = model_dict['stage2_model']

    X_scaled = scaler.transform(X_test)

    pred_aux = np.zeros((len(X_test), len(AUX_COLS)), dtype=np.float32)
    for col_idx, col_name in enumerate(AUX_COLS):
        m = stage1_models.get(col_name)
        if m is not None:
            pred_aux[:, col_idx] = m.predict(X_scaled)

    X_s2 = np.hstack([X_scaled, pred_aux])
    return stage2_model.predict(X_s2)


# ── 直接预测 ───────────────────────────────────────────────────────────────────

def train_predictor_direct(name, X, y, params=None):
    """
    直接预测: X → Q (不使用辅助指标两阶段)

    返回: {'model': model, 'scaler': StandardScaler}
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = create_predictor(name, params)
    model.fit(X_scaled, y)
    return {'model': model, 'scaler': scaler}


def predict_predictor_direct(model_dict, X_test):
    """直接推理"""
    X_scaled = model_dict['scaler'].transform(X_test)
    return model_dict['model'].predict(X_scaled)
