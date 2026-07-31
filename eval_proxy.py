"""
eval_proxy.py — 离线代理评估框架

背景
----
现有 CV-RMSE 系统性乐观：多次实验里离线 CV 越低、线上反而越差
（最典型是 α 扫描：α=1e-3 → CV 122 但线上 255.62；α≈1 → CV 202 但线上 241.86）。
说明现有批次级 GroupKFold 无法反映训练集→测试集的分布偏移。

本脚本提供两样东西：
  1. 一个更"诚实"的离线代理指标 proxy_lobo，尽量消除已知的乐观泄露；
  2. 一个协变量偏移诊断 adversarial AUC，解释 CV 为何乐观。

并支持用已知线上分的配置（α 扫描、PCA vs contrastive）复现，
把结果累积到 output/proxy_eval.csv，供 analyze 子命令计算与线上分的一致性。

proxy_lobo（诚实代理）相对现有 CV 的改动
--------------------------------------
  - 一律留一批次（LeaveOneGroupOut）：每个批次都由"其余全部批次"训练的模型预测，
    折更细、更接近"用全部训练数据预测一个未见批次"的线上设定
    （现有 CV 对 >10 批次的煤种用 GroupKFold(5)，只训练约 80% 批次）。
  - 关闭均值收缩的 OOF 调参（shrink_w≡1.0）：现有 CV 在同一 OOF 上网格搜 w，
    是明显的 in-sample 优化泄露，这里直接用纯模型预测。
  - 每个煤种汇集全部 OOF 批次预测算 pooled RMSE；全局按测试集批次数平方加权
    （与线上 / 现有 CV 口径一致）。

adversarial AUC（协变量偏移诊断）
--------------------------------
  - 对每个煤种训练"训练批次 vs 测试批次"的逻辑回归域分类器，按批次分组交叉验证求 AUC。
  - AUC≈0.5 → 训练/测试同分布，批次级 CV 可信；AUC→1.0 → 偏移大，CV 必然乐观。
"""

import argparse
import os
import sys
import csv
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import roc_auc_score

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR, AUX_COLS
from config import ALPHAS as DEFAULT_ALPHAS
from src.data import load_labels, load_coal_spectra
from src.features import build_feature_matrix, build_feature_matrix_encoder
from src.feature_extractors import get_extractor

PROXY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "proxy_eval.csv")

# 已知线上分（用于一致性检验）；key 与运行时 --tag 对应
# 权威来源：AGENTS.md §实验对比准则（canonical owner）——此处副本必须与该节一致
ONLINE_SCORES = {
    "contrastive_a1":  241.86,   # 对比学习-32, α≥1（当前最优）
    "contrastive_a03": 248.3338, # α=0.3
    "contrastive_a1e3":255.6243, # α≈1e-3
    "pca":             278.50,   # PCA 单路
}


# ── 特征构建 ──────────────────────────────────────────────────────────────────

def build_train_features(train_data, encoder):
    """返回 (X_spec, ctx)；ctx 供测试集复用同一变换。"""
    if encoder is not None:
        X, scaler_hand = build_feature_matrix_encoder(
            train_data, train_data['n_batches'], encoder, fit=True)
        return X, ('enc', encoder, scaler_hand)
    X, ss, pca, sh = build_feature_matrix(train_data, train_data['n_batches'], fit=True)
    return X, ('pca', ss, pca, sh)


def build_test_features(test_data, ctx):
    if ctx[0] == 'enc':
        _, encoder, scaler_hand = ctx
        return build_feature_matrix_encoder(
            test_data, None, encoder, fit=False, scaler_hand=scaler_hand)
    _, ss, pca, sh = ctx
    return build_feature_matrix(
        test_data, n_batches=None, scaler_spec=ss, pca=pca, scaler_hand=sh, fit=False)


# ── 诚实代理指标 ──────────────────────────────────────────────────────────────

def proxy_lobo(train_data, encoder, alphas):
    """留一批次 + 无收缩泄露的 pooled RMSE。返回 (pooled_rmse, n_batches)。"""
    y = train_data['targets']
    aux = train_data['aux']
    groups = train_data['groups']

    X_spec, _ = build_train_features(train_data, encoder)
    logo = list(LeaveOneGroupOut().split(np.zeros(len(y)), None, groups))

    # Stage1: 光谱 → 辅助指标（LOBO OOF）
    aux_oof = np.zeros_like(aux, dtype=np.float32)
    for ci in range(len(AUX_COLS)):
        ya = aux[:, ci]
        if np.isnan(ya).any():
            aux_oof[:, ci] = float(np.nanmean(ya))
            continue
        oof = np.zeros(len(ya))
        for tr, val in logo:
            m = RidgeCV(alphas=alphas)
            m.fit(X_spec[tr], ya[tr])
            oof[val] = m.predict(X_spec[val])
        aux_oof[:, ci] = oof

    # Stage2: [光谱 + OOF 辅助指标] → 发热量（LOBO OOF，无收缩）
    X_s2 = np.hstack([X_spec, aux_oof])
    scaler_s2 = StandardScaler()
    X_s2 = scaler_s2.fit_transform(np.nan_to_num(X_s2))

    bp, bt = [], []
    for tr, val in logo:
        m2 = RidgeCV(alphas=alphas)
        m2.fit(X_s2[tr], y[tr])
        vp = m2.predict(X_s2[val])
        vg = groups[val]
        for bg in np.unique(vg):
            mask = vg == bg
            bt.append(float(y[val][mask][0]))
            bp.append(float(np.median(vp[mask])))

    err = np.array(bp) - np.array(bt)
    return float(np.sqrt(np.mean(err ** 2))), train_data['n_batches']


# ── 协变量偏移诊断 ────────────────────────────────────────────────────────────

def adversarial_auc(train_data, test_data, encoder):
    """训练/测试批次域分类器的分组交叉验证 AUC；越接近 1 偏移越大。"""
    Xtr, ctx = build_train_features(train_data, encoder)
    Xte = build_test_features(test_data, ctx)

    X = np.nan_to_num(np.vstack([Xtr, Xte]))
    y = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))])
    # 分组：训练批次 groups 与测试批次 groups 拼接（测试 group 偏移避免冲突）
    g = np.concatenate([train_data['groups'],
                        test_data['groups'] + train_data['n_batches'] + 1])

    n_groups = len(np.unique(g))
    k = min(5, n_groups)
    if k < 2:
        return float('nan')

    ss = StandardScaler()
    X = ss.fit_transform(X)
    aucs = []
    for tr, val in GroupKFold(n_splits=k).split(X, y, g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[val])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[val])[:, 1]
        aucs.append(roc_auc_score(y[val], p))
    return float(np.mean(aucs)) if aucs else float('nan')


# ── 运行一个配置 ──────────────────────────────────────────────────────────────

def run_config(tag, use_pca, alphas):
    encoder = None if use_pca else get_extractor("contrastive_32")
    extractor = "pca" if use_pca else "contrastive_32"
    print("=" * 64)
    print(f"配置 tag={tag}  extractor={extractor}  alphas={list(alphas)}")
    print("=" * 64)

    label_map, aux_map = load_labels()

    # 测试批次数（全局加权用）
    test_nb = {}
    test_cache = {}
    for ct in COAL_TYPES:
        td = load_coal_spectra(TEST_DIR, ct)
        test_cache[ct] = td
        test_nb[ct] = td['n_batches'] if td else 0

    rmses, weights, aucs, auc_w = [], [], [], []
    for ct in COAL_TYPES:
        tr = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        if tr is None or tr['n_batches'] == 0:
            continue
        rmse, nb = proxy_lobo(tr, encoder, alphas)
        auc = adversarial_auc(tr, test_cache[ct], encoder) if test_cache[ct] else float('nan')
        w = test_nb[ct]
        rmses.append(rmse); weights.append(w)
        if not np.isnan(auc):
            aucs.append(auc); auc_w.append(w)
        print(f"  [{ct:<10}] LOBO批次数={nb:>2}  proxy_RMSE={rmse:7.2f}  "
              f"adv_AUC={auc:.3f}  test_w={w}")

    w = np.array(weights, dtype=np.float32)
    r = np.array(rmses, dtype=np.float32)
    global_proxy = float(np.sqrt(np.average(r ** 2, weights=w)))
    global_auc = float(np.average(aucs, weights=auc_w)) if aucs else float('nan')
    online = ONLINE_SCORES.get(tag, float('nan'))
    print("-" * 64)
    print(f"全局 proxy_RMSE = {global_proxy:.2f}   全局 adv_AUC = {global_auc:.3f}   "
          f"线上 = {online}")

    _append_csv(tag, extractor, list(alphas), global_proxy, global_auc, online)
    return global_proxy, global_auc


def _append_csv(tag, extractor, alphas, proxy, auc, online):
    os.makedirs(os.path.dirname(PROXY_CSV), exist_ok=True)
    new = not os.path.exists(PROXY_CSV)
    with open(PROXY_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["tag", "extractor", "alphas", "proxy_lobo", "adv_auc", "online"])
        wr.writerow([tag, extractor, "|".join(str(a) for a in alphas),
                     f"{proxy:.2f}", f"{auc:.3f}", online])


# ── 一致性分析 ────────────────────────────────────────────────────────────────

def _spearman(a, b):
    """秩相关（无 scipy 依赖）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float('nan')


def analyze():
    """读取 output/proxy_eval.csv，对齐线上分，计算代理分与线上分的一致性。"""
    if not os.path.exists(PROXY_CSV):
        print("无 proxy_eval.csv，先运行各配置。")
        return
    rows = list(csv.DictReader(open(PROXY_CSV, encoding="utf-8")))
    rows = [r for r in rows if r["online"] not in ("", "nan")]
    # 每个 tag 取最后一次
    latest = {}
    for r in rows:
        latest[r["tag"]] = r
    rows = list(latest.values())
    if len(rows) < 2:
        print("有效配置不足 2 个，无法算一致性。当前：", [r["tag"] for r in rows])
        return

    proxy = [float(r["proxy_lobo"]) for r in rows]
    online = [float(r["online"]) for r in rows]
    auc = [float(r["adv_auc"]) for r in rows]

    print("=" * 72)
    print(f"{'tag':<18}{'proxy_lobo':>12}{'adv_auc':>10}{'online':>12}")
    for r in sorted(rows, key=lambda x: float(x["online"])):
        print(f"{r['tag']:<18}{float(r['proxy_lobo']):>12.2f}"
              f"{float(r['adv_auc']):>10.3f}{float(r['online']):>12.2f}")
    print("-" * 72)
    print(f"Spearman(proxy_lobo, online) = {_spearman(proxy, online):+.3f}  "
          f"(越接近 +1 越一致)")

    # α 单调性专项（只看 contrastive α 三点）
    amap = {t: r for t, r in latest.items()}
    a_tags = ["contrastive_a1e3", "contrastive_a03", "contrastive_a1"]
    if all(t in amap for t in a_tags):
        pv = [float(amap[t]["proxy_lobo"]) for t in a_tags]
        ov = [float(amap[t]["online"]) for t in a_tags]
        print("\nα 单调性检验（α: 1e-3 → 0.3 → 1.0）")
        print(f"  线上       : {ov[0]:.2f} -> {ov[1]:.2f} -> {ov[2]:.2f}  "
              f"({'单调下降[OK]' if ov[0] > ov[1] > ov[2] else '非单调'})")
        print(f"  proxy_lobo : {pv[0]:.2f} -> {pv[1]:.2f} -> {pv[2]:.2f}  "
              f"({'单调下降[OK] 与线上同向' if pv[0] > pv[1] > pv[2] else '未复现线上方向[FAIL]'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="配置标签，用于对齐线上分")
    ap.add_argument("--pca", action="store_true", help="用 PCA 路径（默认 contrastive_32）")
    ap.add_argument("--alpha", type=float, default=None,
                    help="强制单一 Ridge α（不填则用 config.ALPHAS）")
    ap.add_argument("--analyze", action="store_true", help="仅做一致性分析")
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return

    alphas = [args.alpha] if args.alpha is not None else list(DEFAULT_ALPHAS)
    tag = args.tag or ("pca" if args.pca else "contrastive_a1")
    run_config(tag, args.pca, alphas)


if __name__ == "__main__":
    main()
