"""
光谱数据增强模块

所有增强操作均在**批次内**进行，增强样本与原样本共享同一 group index，
确保 GroupKFold/LOOCV 评估结构不受影响，防止数据泄漏。

增强策略:
  noise    : Gaussian additive noise (模拟测量噪声)
  mixup    : 同批次内两谱线性插值 (隐式正则化)
  jitter   : 随机振幅缩放 + 小噪声 (模拟激光能量波动)
  combined : noise + jitter + mixup 依次应用
"""

import numpy as np

# ── 单谱增强基元 ──────────────────────────────────────────────────────────────

def _add_noise(inten: np.ndarray, rng: np.random.Generator,
               noise_factor: float = 0.02) -> np.ndarray:
    """高斯加性噪声"""
    noise = rng.normal(0, noise_factor * inten.std() + 1e-8, inten.shape)
    return np.maximum(inten + noise.astype(np.float32), 0.0)


def _jitter(inten: np.ndarray, rng: np.random.Generator,
            jitter_range: tuple = (0.9, 1.1),
            noise_factor: float = 0.005) -> np.ndarray:
    """随机振幅缩放 + 微量噪声"""
    scale = rng.uniform(*jitter_range)
    noise = rng.normal(0, noise_factor * inten.std() + 1e-8, inten.shape)
    return np.maximum(inten * scale + noise.astype(np.float32), 0.0)


def _mixup_spectra(spec_a: tuple, spec_b: tuple,
                   rng: np.random.Generator,
                   alpha: float = 0.5) -> tuple:
    """两条光谱的线性插值"""
    wl1, inten1 = spec_a
    wl2, inten2 = spec_b
    # 波长对齐：截断到较短长度
    min_len = min(len(inten1), len(inten2))
    lam = rng.beta(alpha, alpha)
    inten_mix = lam * inten1[:min_len] + (1 - lam) * inten2[:min_len]
    return (wl1[:min_len], inten_mix.astype(np.float32))


# ── 批次级别 Mixup ────────────────────────────────────────────────────────────

def _batch_mixup(spectra_batch: list, rng: np.random.Generator,
                 alpha: float = 0.5, aug_factor: int = 1) -> list:
    """对同一批次内的光谱做 mixup，返回增强光谱列表"""
    n = len(spectra_batch)
    if n < 2:
        return []
    mixed = []
    for _ in range(aug_factor):
        i, j = rng.integers(0, n), rng.integers(0, n)
        while j == i and n > 1:
            j = rng.integers(0, n)
        mixed.append(_mixup_spectra(spectra_batch[i], spectra_batch[j], rng, alpha))
    return mixed


# ── 主入口 ────────────────────────────────────────────────────────────────────

def augment_data(data: dict, strategy: str = 'noise',
                 aug_factor: int = 1, seed: int = 42,
                 alpha: float = 0.5,             # mixup Beta α 参数
                 noise_factor: float = 0.02,     # noise/jitter 噪声强度
                 jitter_min: float = 0.9,        # jitter 缩放下限
                 jitter_max: float = 1.1) -> dict:  # jitter 缩放上限
    """
    对训练数据做光谱级数据增强。

    增强光谱与原光谱共享同一 group index，GroupKFold 分割时
    同批次的增强数据不会泄漏到其他折。

    Parameters
    ----------
    data : dict
        load_coal_spectra() 返回的训练数据字典
    strategy : str
        增强策略: 'noise', 'mixup', 'jitter', 'combined'
    aug_factor : int
        每条原始光谱生成的增强副本数
    seed : int
        随机种子

    Returns
    -------
    dict
        增强后的数据字典（含原始 + 增强样本）
    """
    rng = np.random.default_rng(seed)
    spectra = data['spectra']
    n_orig = len(spectra)

    # 新数据容器：先保留原始数据
    new_spectra = list(spectra)
    new_names = list(data['names'])
    new_groups = list(data['groups'])

    has_target = data['targets'] is not None
    has_aux = data['aux'] is not None
    new_targets = list(data['targets']) if has_target else None
    new_aux = list(data['aux']) if has_aux else None

    if strategy in ('noise', 'jitter'):
        # ── 逐光谱增强 ──
        for i in range(n_orig):
            wl, inten = spectra[i]
            for _ in range(aug_factor):
                if strategy == 'noise':
                    aug_inten = _add_noise(inten, rng, noise_factor=noise_factor)
                else:
                    aug_inten = _jitter(inten, rng,
                                         jitter_range=(jitter_min, jitter_max),
                                         noise_factor=noise_factor)

                new_spectra.append((wl, aug_inten))
                new_names.append(data['names'][i])
                new_groups.append(data['groups'][i])
                if has_target:
                    new_targets.append(data['targets'][i])
                if has_aux:
                    new_aux.append(data['aux'][i])

    elif strategy == 'mixup':
        # ── 批次级 Mixup ──
        batch_indices = {}
        for i, g in enumerate(data['groups']):
            batch_indices.setdefault(int(g), []).append(i)

        for g, indices in batch_indices.items():
            batch_spectra = [spectra[i] for i in indices]
            name = data['names'][indices[0]]
            target = data['targets'][indices[0]] if has_target else None
            aux = data['aux'][indices[0]] if has_aux else None

            mixed = _batch_mixup(batch_spectra, rng, alpha=alpha, aug_factor=aug_factor)
            for m_spec in mixed:
                new_spectra.append(m_spec)
                new_names.append(name)
                new_groups.append(g)
                if has_target:
                    new_targets.append(target)
                if has_aux:
                    new_aux.append(aux)

    elif strategy == 'combined':
        # ── 组合增强: noise + jitter(每光谱) + mixup(批次) ──
        # 先做逐光谱增强
        n_per_spec_aug = max(1, aug_factor // 2)
        for i in range(n_orig):
            wl, inten = spectra[i]
            for _ in range(n_per_spec_aug):
                aug_inten = _add_noise(inten, rng, noise_factor=noise_factor * 0.5)
                aug_inten = _jitter(aug_inten, rng,
                                     jitter_range=(jitter_min, jitter_max),
                                     noise_factor=noise_factor * 0.3)
                new_spectra.append((wl, aug_inten))
                new_names.append(data['names'][i])
                new_groups.append(data['groups'][i])
                if has_target:
                    new_targets.append(data['targets'][i])
                if has_aux:
                    new_aux.append(data['aux'][i])

        # 再做 mixup
        batch_indices = {}
        for i, g in enumerate(data['groups']):
            batch_indices.setdefault(int(g), []).append(i)

        n_mixup = max(1, aug_factor - n_per_spec_aug)
        for g, indices in batch_indices.items():
            batch_spectra = [spectra[i] for i in indices]
            name = data['names'][indices[0]]
            target = data['targets'][indices[0]] if has_target else None
            aux = data['aux'][indices[0]] if has_aux else None

            mixed = _batch_mixup(batch_spectra, rng, alpha=alpha, aug_factor=n_mixup)
            for m_spec in mixed:
                new_spectra.append(m_spec)
                new_names.append(name)
                new_groups.append(g)
                if has_target:
                    new_targets.append(target)
                if has_aux:
                    new_aux.append(aux)
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

    # ── 构建增强后数据字典 ──
    result = dict(data)  # shallow copy
    result['spectra'] = new_spectra
    result['names'] = new_names
    result['groups'] = np.array(new_groups, dtype=np.int32)
    # 批次数不变（增强样本不增加新批次）
    result['n_batches'] = data['n_batches']

    if has_target:
        result['targets'] = np.array(new_targets, dtype=np.float64)
    if has_aux:
        result['aux'] = np.array(new_aux, dtype=np.float32)

    # 清除预计算特征（需重新计算）
    n_total = len(new_spectra)
    result['stats'] = [None] * n_total
    result['labs'] = [None] * n_total
    result['lrel'] = [None] * n_total
    result['rats'] = [None] * n_total

    return result
