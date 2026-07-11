"""
特征工程模块

LIBS 光谱特征分三层:
  1. 统计特征   — 均值/方差/偏度/峰度/熵/分位数/导数
  2. 谱线特征   — 关键元素谱线积分（绝对值 & 相对归一化）
  3. 物理比值   — 可燃元素/灰分比，H/C 比等

三层拼接后进行 PCA 降维 + 标准化，作为 Ridge 模型的输入。
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import KEY_LINES, N_PCA_MAX, RANDOM_STATE


# ── 谱线积分 ──────────────────────────────────────────────────────────────────

def line_integral(wl, inten, center_nm, halfwin_nm):
    """
    对 [center-halfwin, center+halfwin] 窗口内的强度求和。
    用于提取特定元素的发射谱线强度。
    """
    mask = (wl >= center_nm - halfwin_nm) & (wl <= center_nm + halfwin_nm)
    return float(inten[mask].sum()) if mask.sum() > 0 else 0.0


# ── 单条光谱特征 ──────────────────────────────────────────────────────────────

def spectrum_features(wl, inten):
    """
    从一条光谱提取三层特征。

    返回:
        inorm  (D,)   : 强度归一化（用于 PCA）
        stats  (17,)  : 统计特征
        labs   (11,)  : 谱线绝对积分
        lrel   (11,)  : 谱线相对积分
        rats   (4,)   : 物理比值
    """
    total      = inten.sum() + 1e-8
    inten_norm = inten / total          # 归一化强度（消除激光能量波动）
    deriv      = np.diff(inten_norm)    # 一阶差分
    deriv2     = np.diff(inten_norm, 2) # 二阶差分

    # 统计特征: 描述整体光谱形状
    stats = np.array([
        inten.mean(), inten.std(), inten.max(), np.log1p(total),
        inten_norm.mean(), inten_norm.std(),
        float(skew(inten_norm)),
        float(kurtosis(inten_norm)),
        float(scipy_entropy(inten_norm + 1e-12)),
        np.percentile(inten, 10), np.percentile(inten, 25),
        np.percentile(inten, 75), np.percentile(inten, 90),
        deriv.std(),  float(np.abs(deriv).mean()),
        deriv2.std(), float(np.abs(deriv2).mean()),
    ], dtype=np.float32)

    # 谱线特征: 物理意义最强的部分
    line_names = list(KEY_LINES.keys())
    labs = np.array(
        [line_integral(wl, inten, c, h) for _, (c, h) in KEY_LINES.items()],
        dtype=np.float32
    )
    lrel = labs / total   # 相对强度（消除光谱总能量差异）

    # 物理比值: 可燃元素 vs 灰分元素
    # 索引: C=0, H=1, O=2, N=3 | Ca=4, Ca2=5, Mg=6, Al=7, Si=8, Fe=9, Na=10
    comb_sum = labs[[0, 1, 2, 3]].sum()                        # 可燃元素
    ash_sum  = labs[[4, 5, 6, 7, 8, 9]].sum() + 1e-8          # 灰分元素
    rats = np.array([
        comb_sum / ash_sum,                 # 可燃/灰分 (发热量代理)
        labs[1] / ash_sum,                  # H/灰分
        labs[0] / ash_sum,                  # C/灰分
        labs[1] / (labs[0] + 1e-8),        # H/C
    ], dtype=np.float32)

    return inten_norm, stats, labs, lrel, rats


# ── 批量特征计算 ──────────────────────────────────────────────────────────────

def compute_features(data):
    """
    对 data_dict 中所有光谱计算特征，原地填充 stats/labs/lrel/rats 字段。
    同时返回归一化光谱矩阵（用于 PCA）。
    """
    raw_spectra = data['spectra']
    inorms, stats_list, labs_list, lrel_list, rats_list = [], [], [], [], []

    for wl, inten in raw_spectra:
        inorm, stats, labs, lrel, rats = spectrum_features(wl, inten)
        inorms.append(inorm)
        stats_list.append(stats)
        labs_list.append(labs)
        lrel_list.append(lrel)
        rats_list.append(rats)

    # 裁剪到最短公共波长（不同仪器/文件行数可能略有差异）
    min_len = min(len(s) for s in inorms)
    inorm_mat = np.array([s[:min_len] for s in inorms], dtype=np.float32)

    data['stats'] = np.array(stats_list)
    data['labs']  = np.array(labs_list)
    data['lrel']  = np.array(lrel_list)
    data['rats']  = np.array(rats_list)

    return inorm_mat


# ── PCA + 拼接 ────────────────────────────────────────────────────────────────

def build_feature_matrix(data, n_batches,
                         scaler_spec=None, pca=None, scaler_hand=None, fit=True):
    """
    构建最终特征矩阵: [PCA(归一化光谱)] + [手工特征]。

    fit=True  : 用训练数据拟合 scaler/PCA，返回 (X, scaler_spec, pca, scaler_hand)
    fit=False : 用已拟合的 scaler/PCA 变换测试数据，返回 X

    参数:
        n_batches: 批次数，用于限制 PCA 维度（不能超过样本数-1）
    """
    inorm_mat = compute_features(data)

    if fit:
        scaler_spec = StandardScaler()
        spec_scaled = scaler_spec.fit_transform(inorm_mat)
        n_pca = min(N_PCA_MAX, n_batches - 1, inorm_mat.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
        spec_pca = pca.fit_transform(spec_scaled)
    else:
        spec_pca = pca.transform(scaler_spec.transform(inorm_mat))

    # 拼接手工特征
    hand_feats = np.hstack([data['stats'], data['labs'], data['lrel'], data['rats']])
    X = np.hstack([spec_pca, hand_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        scaler_hand = StandardScaler()
        X = scaler_hand.fit_transform(X)
        return X, scaler_spec, pca, scaler_hand
    else:
        return scaler_hand.transform(X)
