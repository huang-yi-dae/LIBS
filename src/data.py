"""
数据加载模块

职责:
  - 读取单条光谱 CSV（跳过4行元数据+1行表头）
  - 读取标签 Excel（发热量 + 辅助指标）
  - 遍历煤种目录，按批次收集所有光谱
"""

import os
import glob
import numpy as np
import pandas as pd

from config import LABEL_DIR, AUX_COLS


def read_spectrum_csv(filepath):
    """
    解析单条光谱文件。
    格式: 前5行为元数据/表头，之后每行 "波长,强度," (含尾逗号)。

    返回:
        wl    (np.ndarray float32): 波长数组
        inten (np.ndarray float32): 强度数组
        两者均为 None 表示读取失败
    """
    wl, inten = [], []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i < 5:          # 跳过元数据
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    try:
                        wl.append(float(parts[0]))
                        inten.append(float(parts[1]))
                    except ValueError:
                        continue
    except Exception:
        return None, None

    if not wl:
        return None, None
    return np.array(wl, dtype=np.float32), np.array(inten, dtype=np.float32)


def load_labels():
    """
    读取训练集标签（所有煤种的 Excel 文件）。

    返回:
        label_map: {(煤种名, 批次日期) -> 发热量 float}
        aux_map:   {(煤种名, 批次日期) -> {辅助指标名: float}}
    """
    label_map = {}
    aux_map   = {}

    for xlsx in glob.glob(os.path.join(LABEL_DIR, "*.xlsx")):
        coal_type = os.path.splitext(os.path.basename(xlsx))[0]
        df = pd.read_excel(xlsx)
        df.columns = [c.strip() for c in df.columns]
        df['名称'] = df['名称'].astype(str).str.strip()

        for _, row in df.iterrows():
            key = (coal_type, row['名称'])
            label_map[key] = float(row['发热量(Q)'])
            aux_map[key]   = {c: float(row.get(c, np.nan)) for c in AUX_COLS}

    return label_map, aux_map


def load_coal_spectra(data_root, coal_type, label_map=None, aux_map=None):
    """
    读取某煤种下所有批次的光谱数据。

    多对一对齐策略: 同一批次内所有光谱共享同一发热量标签，
    用 groups 数组标记批次归属，供 GroupKFold 防止数据泄露。

    参数:
        data_root:  训练集或测试集的根目录
        coal_type:  煤种名（与目录名一致）
        label_map:  训练时传入，测试时传 None
        aux_map:    同上

    返回 dict（键见下方），所有字段均对齐到光谱条数:
        spectra  : list of (wavelength, intensity) tuples
        stats    : (N, 17)  统计特征（由 features.py 填充）
        labs     : (N, 11)  谱线绝对积分
        lrel     : (N, 11)  谱线相对积分
        rats     : (N, 4)   物理比值特征
        names    : list[str]  批次文件夹名
        targets  : (N,) or None
        aux      : (N, 4) or None
        groups   : (N,) int  批次索引
        n_batches: int
    """
    coal_dir = os.path.join(data_root, coal_type)
    if not os.path.isdir(coal_dir):
        return None

    raw_spectra = []
    stats_l, labs_l, lrel_l, rats_l = [], [], [], []
    names, targets, aux_targets, groups = [], [], [], []
    batch_idx = 0

    for batch_folder in sorted(os.listdir(coal_dir)):
        batch_dir = os.path.join(coal_dir, batch_folder)
        if not os.path.isdir(batch_dir):
            continue

        # 批次日期 = 文件夹名去掉煤种前缀
        date_str = batch_folder.replace(coal_type, '', 1).strip()

        q, aux = None, None
        if label_map is not None:
            key = (coal_type, date_str)
            if key not in label_map:
                continue
            q   = label_map[key]
            aux = aux_map.get(key, {c: np.nan for c in AUX_COLS}) if aux_map else None

        csvs = glob.glob(os.path.join(batch_dir, "*.csv"))
        batch_has_data = False
        for f in csvs:
            wl, inten = read_spectrum_csv(f)
            if wl is None:
                continue
            # 先存原始光谱，features.py 负责计算特征
            raw_spectra.append((wl, inten))
            stats_l.append(None)    # 占位，由 features.py 填充
            labs_l.append(None)
            lrel_l.append(None)
            rats_l.append(None)
            names.append(batch_folder)
            if q is not None:
                targets.append(q)
            if aux is not None:
                aux_targets.append([aux[c] for c in AUX_COLS])
            groups.append(batch_idx)
            batch_has_data = True

        if batch_has_data:
            batch_idx += 1

    if not raw_spectra:
        return None

    return {
        'spectra':   raw_spectra,           # list of (wl, inten) tuples
        'stats':     stats_l,               # 填充前为 None 列表
        'labs':      labs_l,
        'lrel':      lrel_l,
        'rats':      rats_l,
        'names':     names,
        'targets':   np.array(targets)      if targets      else None,
        'aux':       np.array(aux_targets)  if aux_targets  else None,
        'groups':    np.array(groups),
        'n_batches': batch_idx,
    }
