"""
eval_combos.py — 组合方向离线评估（复用 proxy_lobo 的诚实口径）

目的
----
在 eval_proxy.py 验证「proxy_lobo 与线上秩相关 +0.80」之后，本脚本把 README
「候选方向」里的几条**方差抑制/校准类**方向落地为可自由组合的开关，用同一套
留一批次(LOBO) + 测试批次数平方加权的 pooled-RMSE 口径评估、排序，选出值得
线上验证的组合。

已知诊断：adv_AUC≈0.54（近 0.5）说明线上退化并非特征协变量偏移，而是目标条件
偏移 + 26 个测试批次的小样本方差。故本脚本聚焦「降方差」而非「换特征」。

可组合开关
----------
  --agg {median,mean,trimmed}   批次内多光谱聚合方式（默认 median = 现状）
  --aux-filter                  剔除 Stage1 中 OOF R²<0 的辅助列（比均值还差 → 纯噪声）
  --shrink-scan                 扫描全局收缩权重 w（折内训练批次均值做锚点，无泄漏）
  --pooled-stage1               Stage1 辅助模型跨煤种联合训练（共享特征空间）
  --tag NAME                    结果写入 output/combos_eval.csv 的标签
  --analyze                     读取 combos_eval.csv，按 proxy 排序展示

设计要点（与 proxy_lobo 保持口径一致，保证可比）
  - 一律 LeaveOneGroupOut（留一批次）。
  - 收缩锚点用**该折训练批次**的发热量均值（不是全量均值），杜绝 in-sample 泄漏。
  - w 为全局超参（所有煤种共用一个 w），扫描后报告；采纳仍需线上验证。
  - 每煤种 pooled RMSE，全局按测试集批次数平方加权，与线上口径一致。
"""

import argparse
import os
import sys
import csv
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR, AUX_COLS
from config import ALPHAS as DEFAULT_ALPHAS
from src.data import load_labels, load_coal_spectra
from src.features import compute_features
from src.feature_extractors import get_extractor
from src.pretrain import DEVICE as PT_DEVICE

COMBOS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "combos_eval.csv")

# 收缩权重扫描网格（w=1.0 即不收缩）
SHRINK_GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]

# 当前最优离线基准（contrastive_a1 的 proxy_lobo，见 eval_proxy 结论）
# 权威来源：AGENTS.md §实验对比准则（canonical owner）——此处副本必须与该节一致
PROXY_BASELINE = 161.70


# ── 原始特征（未按煤种标准化，供跨煤种共享用）────────────────────────────────

def raw_feats(coal_data, encoder):
    """返回未标准化的 [encoder隐变量 + 手工特征] 矩阵 (N, D)。"""
    inorm = compute_features(coal_data).astype(np.float32)
    encoder = encoder.to(PT_DEVICE)
    encoder.eval()
    with torch.no_grad():
        latent = encoder.encode(torch.from_numpy(inorm).to(PT_DEVICE)).cpu().numpy()
    hand = np.hstack([coal_data['stats'], coal_data['labs'],
                      coal_data['lrel'], coal_data['rats']])
    X = np.hstack([latent, hand])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ── 批次聚合 ──────────────────────────────────────────────────────────────────

def _agg(vals, how, trim_pct=10.0):
    """批次内多光谱预测聚合。

    how:
        median  — 现状（稳健，但丢弃分布信息）
        mean    — 全均值（对离群谱敏感）
        trimmed — 去掉上下 trim_pct% 后取均值（去尾均值）
        winsor  — 把上下 trim_pct% 截断到分位点后取均值（缩尾均值，保留样本量）
        avg3    — median / trimmed / mean 三者的均值（聚合器级集成，进一步降方差）
    """
    v = np.asarray(vals, float)
    if how == 'mean':
        return float(v.mean())
    if how == 'trimmed':
        if len(v) >= 5:
            lo, hi = np.percentile(v, trim_pct), np.percentile(v, 100.0 - trim_pct)
            keep = v[(v >= lo) & (v <= hi)]
            return float(keep.mean()) if len(keep) else float(np.median(v))
        return float(np.median(v))
    if how == 'winsor':
        if len(v) >= 5:
            lo, hi = np.percentile(v, trim_pct), np.percentile(v, 100.0 - trim_pct)
            return float(np.clip(v, lo, hi).mean())
        return float(np.median(v))
    if how == 'avg3':
        return float(np.mean([_agg(v, 'median'), _agg(v, 'trimmed', trim_pct),
                              _agg(v, 'mean')]))
    return float(np.median(v))  # default median


# ── 单煤种 LOBO 评估（返回批次级 pred/true/fold_mean）─────────────────────────

def lobo_batches(X, y, aux, groups, alphas, agg='median', aux_filter=False,
                 aux_oof_ext=None, trim_pct=10.0):
    """
    留一批次评估，返回该煤种的批次级 [(pred, true, fold_train_mean), ...]。

    参数:
        aux_oof_ext: 若给定 (N, 4) 的跨煤种 Stage1 OOF 辅助预测，则直接使用
                     （用于 pooled_stage1）；否则本煤种内 LOBO 生成 Stage1 OOF。
        aux_filter:  剔除 OOF R²<0 的辅助列。
    """
    logo = list(LeaveOneGroupOut().split(np.zeros(len(y)), None, groups))

    # ── Stage1: 光谱 → 辅助指标 OOF ───────────────────────────────
    if aux_oof_ext is not None:
        aux_oof = aux_oof_ext
    else:
        aux_oof = np.zeros_like(aux, dtype=np.float32)
        for ci in range(len(AUX_COLS)):
            ya = aux[:, ci]
            if np.isnan(ya).any():
                aux_oof[:, ci] = float(np.nanmean(ya))
                continue
            oof = np.zeros(len(ya))
            for tr, val in logo:
                m = RidgeCV(alphas=alphas)
                m.fit(X[tr], ya[tr])
                oof[val] = m.predict(X[val])
            aux_oof[:, ci] = oof

    # 辅助列质量筛选（OOF R²<0 → 丢弃）
    keep_cols = list(range(len(AUX_COLS)))
    dropped = []
    if aux_filter:
        keep_cols = []
        for ci in range(len(AUX_COLS)):
            ya = aux[:, ci]
            if np.isnan(ya).any():
                continue  # 缺失列本就是常数填充，无信息，丢弃
            sse = float(np.sum((aux_oof[:, ci] - ya) ** 2))
            sst = float(np.sum((ya - ya.mean()) ** 2)) + 1e-12
            r2 = 1.0 - sse / sst
            if r2 >= 0.0:
                keep_cols.append(ci)
            else:
                dropped.append((AUX_COLS[ci], r2))

    aux_used = aux_oof[:, keep_cols] if keep_cols else np.zeros((len(y), 0), np.float32)

    # ── Stage2: [光谱 + 保留的OOF辅助] → 发热量 ───────────────────
    X_s2 = np.hstack([X, aux_used]) if aux_used.shape[1] else X.copy()
    scaler = StandardScaler()
    X_s2 = scaler.fit_transform(np.nan_to_num(X_s2))

    batches = []
    for tr, val in logo:
        m2 = RidgeCV(alphas=alphas)
        m2.fit(X_s2[tr], y[tr])
        vp = m2.predict(X_s2[val])
        vg = groups[val]
        fold_mean = float(y[tr].mean())  # 折内训练批次均值（无泄漏锚点）
        for bg in np.unique(vg):
            mask = vg == bg
            batches.append((_agg(vp[mask], agg, trim_pct), float(y[val][mask][0]), fold_mean))
    return batches, dropped


def coal_rmse(batches, w=1.0):
    """给定收缩权重 w，算该煤种 pooled RMSE。"""
    pred = np.array([w * p + (1 - w) * fm for p, t, fm in batches])
    true = np.array([t for p, t, fm in batches])
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# ── 跨煤种 Stage1 OOF（pooled）────────────────────────────────────────────────

def pooled_stage1_oof(coal_raw, coal_y_aux, alphas):
    """
    Stage1 辅助模型跨煤种联合训练：对每个煤种的每个留出批次，用「其余全部煤种 +
    本煤种其余批次」训练辅助模型来预测。返回 {煤种: (N,4) OOF}。

    共享特征空间：在全部训练煤种的原始特征上 fit 一个 StandardScaler。
    """
    order = list(coal_raw.keys())
    all_raw = np.vstack([coal_raw[c] for c in order])
    scaler = StandardScaler().fit(np.nan_to_num(all_raw))
    coal_X = {c: scaler.transform(np.nan_to_num(coal_raw[c])) for c in order}

    # 每煤种一个全局 batch 偏移，拼一个统一 groups
    offsets, cursor = {}, 0
    for c in order:
        offsets[c] = cursor
        cursor += int(coal_groups[c].max()) + 1

    out = {c: np.zeros_like(coal_y_aux[c], dtype=np.float32) for c in order}
    for ci in range(len(AUX_COLS)):
        # 汇总所有煤种该列的标签与特征
        Xs, ys, gs, owner = [], [], [], []
        for c in order:
            ya = coal_y_aux[c][:, ci]
            Xs.append(coal_X[c]); ys.append(ya)
            gs.append(coal_groups[c] + offsets[c])
            owner.append(np.full(len(ya), order.index(c)))
        Xall = np.vstack(Xs); yall = np.concatenate(ys)
        gall = np.concatenate(gs); oall = np.concatenate(owner)

        if np.isnan(yall).any():
            # 该列在某些煤种缺失 → 逐煤种用各自均值填充
            for c in order:
                ya = coal_y_aux[c][:, ci]
                out[c][:, ci] = float(np.nanmean(ya)) if not np.isnan(ya).all() else 0.0
            continue

        for c in order:
            gco = coal_groups[c] + offsets[c]
            for bg in np.unique(gco):
                held = (gall == bg)
                train = ~held
                m = RidgeCV(alphas=alphas)
                m.fit(Xall[train], yall[train])
                idx_local = (coal_groups[c] + offsets[c]) == bg
                out[c][idx_local, ci] = m.predict(coal_X[c][idx_local])
    return coal_X, out


# ── 运行一个组合配置 ──────────────────────────────────────────────────────────

coal_groups = {}  # 模块级缓存：{煤种: groups}


def run_combo(tag, agg, aux_filter, shrink_scan, pooled_stage1, alphas, trim_pct=10.0):
    print("=" * 70)
    print(f"配置 tag={tag}  agg={agg}  trim_pct={trim_pct}  aux_filter={aux_filter}  "
          f"shrink_scan={shrink_scan}  pooled_stage1={pooled_stage1}")
    print("=" * 70)

    encoder = get_extractor("contrastive_32")
    label_map, aux_map = load_labels()

    # 测试批次数（全局加权）
    test_nb = {}
    for ct in COAL_TYPES:
        td = load_coal_spectra(TEST_DIR, ct)
        test_nb[ct] = td['n_batches'] if td else 0

    # 加载训练数据 + 原始特征
    coal_raw, coal_y, coal_y_aux = {}, {}, {}
    global coal_groups
    coal_groups = {}
    for ct in COAL_TYPES:
        tr = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        if tr is None or tr['n_batches'] == 0:
            continue
        coal_raw[ct] = raw_feats(tr, encoder)
        coal_y[ct] = tr['targets']
        coal_y_aux[ct] = tr['aux']
        coal_groups[ct] = tr['groups']

    # pooled Stage1：跨煤种共享特征 + 联合 OOF
    aux_oof_map = None
    if pooled_stage1:
        coal_X, aux_oof_map = pooled_stage1_oof(coal_raw, coal_y_aux, alphas)
    else:
        # 每煤种各自标准化
        coal_X = {c: StandardScaler().fit_transform(np.nan_to_num(coal_raw[c]))
                  for c in coal_raw}

    # 逐煤种 LOBO
    coal_batches, weights, order = {}, [], []
    for ct in coal_raw:
        ext = aux_oof_map[ct] if aux_oof_map is not None else None
        batches, dropped = lobo_batches(
            coal_X[ct], coal_y[ct], coal_y_aux[ct], coal_groups[ct],
            alphas, agg=agg, aux_filter=aux_filter, aux_oof_ext=ext, trim_pct=trim_pct)
        coal_batches[ct] = batches
        weights.append(test_nb[ct]); order.append(ct)
        drp = ("  丢弃辅助:" + ",".join(f"{n}({r:+.2f})" for n, r in dropped)) if dropped else ""
        print(f"  [{ct:<10}] 批次={len(batches):>2}  test_w={test_nb[ct]}{drp}")

    w = np.array(weights, float)

    def global_at(ww):
        r = np.array([coal_rmse(coal_batches[c], ww) for c in order])
        return float(np.sqrt(np.average(r ** 2, weights=w)))

    if shrink_scan:
        print("-" * 70)
        print("  收缩权重扫描 (w: 模型权重, 1-w 向折内均值收缩)")
        best_w, best_g = 1.0, float('inf')
        for ww in SHRINK_GRID:
            g = global_at(ww)
            flag = ""
            if g < best_g:
                best_g, best_w = g, ww
            print(f"    w={ww:.2f}  global_proxy={g:7.2f}")
        print(f"  >> 最优 w={best_w:.2f}  proxy={best_g:.2f}")
        global_proxy, chosen_w = best_g, best_w
    else:
        global_proxy, chosen_w = global_at(1.0), 1.0

    delta = global_proxy - PROXY_BASELINE
    print("-" * 70)
    print(f"全局 proxy_lobo = {global_proxy:.2f}  (基准 {PROXY_BASELINE:.2f}, "
          f"{'↓' if delta < 0 else '↑'}{abs(delta):.2f})  最优w={chosen_w:.2f}")

    _append(tag, agg, aux_filter, shrink_scan, pooled_stage1, chosen_w, global_proxy)
    return global_proxy


def _append(tag, agg, aux_filter, shrink_scan, pooled, w, proxy):
    os.makedirs(os.path.dirname(COMBOS_CSV), exist_ok=True)
    new = not os.path.exists(COMBOS_CSV)
    with open(COMBOS_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["tag", "agg", "aux_filter", "shrink_scan",
                         "pooled_stage1", "best_w", "proxy_lobo"])
        wr.writerow([tag, agg, int(aux_filter), int(shrink_scan),
                     int(pooled), f"{w:.2f}", f"{proxy:.2f}"])


def analyze():
    if not os.path.exists(COMBOS_CSV):
        print("无 combos_eval.csv，先运行各组合。")
        return
    rows = list(csv.DictReader(open(COMBOS_CSV, encoding="utf-8")))
    latest = {}
    for r in rows:
        latest[r["tag"]] = r
    rows = sorted(latest.values(), key=lambda x: float(x["proxy_lobo"]))
    print("=" * 84)
    print(f"{'tag':<22}{'agg':>8}{'auxF':>6}{'shrink':>8}{'pooled':>8}"
          f"{'best_w':>8}{'proxy':>9}")
    print("-" * 84)
    for r in rows:
        print(f"{r['tag']:<22}{r['agg']:>8}{r['aux_filter']:>6}{r['shrink_scan']:>8}"
              f"{r['pooled_stage1']:>8}{r['best_w']:>8}{float(r['proxy_lobo']):>9.2f}")
    print("-" * 84)
    print(f"基准 proxy_lobo = {PROXY_BASELINE:.2f}（contrastive_a1，线上 241.86）")
    print("注：proxy 仅方向性参考（与线上 Spearman +0.80），最终以线上验证为准。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--agg", choices=["median", "mean", "trimmed", "winsor", "avg3"],
                    default="median")
    ap.add_argument("--trim-pct", type=float, default=10.0,
                    help="trimmed/winsor/avg3 的上下截断百分比")
    ap.add_argument("--aux-filter", action="store_true")
    ap.add_argument("--shrink-scan", action="store_true")
    ap.add_argument("--pooled-stage1", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return

    tag = args.tag or "combo"
    run_combo(tag, args.agg, args.aux_filter, args.shrink_scan,
              args.pooled_stage1, list(DEFAULT_ALPHAS), trim_pct=args.trim_pct)


if __name__ == "__main__":
    main()
