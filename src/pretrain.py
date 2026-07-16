"""
预训练编码器模块 — AE / MAE / Contrastive Learning

所有模型以归一化光谱 (inorm_mat) 为输入，输出低维隐变量。
训练时不使用标签，属于自监督/无监督学习。

每个模型提供:
  - train(data_dict, ...)     → 训练并返回 encoder
  - extract(encoder, data)    → 提取隐变量特征矩阵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

from config import RANDOM_STATE
from src.features import compute_features


# ── 设备 ──────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据集构建
# ══════════════════════════════════════════════════════════════════════════════

def build_pretrain_dataset(data_dicts, val_ratio=0.15):
    """
    将所有煤种的归一化光谱合并，划分训练/验证集。

    参数:
        data_dicts: list[dict], 每个煤种的 data dict（含 targets）
        val_ratio:  验证集比例（按批次划分，防止数据泄露）

    返回:
        train_loader, val_loader, input_dim
    """
    all_spectra, all_groups, all_targets = [], [], []
    offset = 0

    for data in data_dicts:
        if data is None:
            continue
        inorm = compute_features(data)  # 填充手工特征 + 返回归一化光谱
        all_spectra.append(inorm)
        # groups 跨煤种不重叠
        all_groups.append(data['groups'] + offset)
        offset += data['n_batches']
        if data['targets'] is not None:
            all_targets.append(data['targets'])

    X = np.vstack(all_spectra).astype(np.float32)
    groups = np.concatenate(all_groups)
    y = np.concatenate(all_targets) if all_targets else None

    # 按批次划分验证集（避免同批次数据泄露）
    unique_groups = np.unique(groups)
    train_g, val_g = train_test_split(
        unique_groups, test_size=val_ratio, random_state=RANDOM_STATE
    )
    train_idx = np.isin(groups, train_g)
    val_idx   = np.isin(groups, val_g)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx] if y is not None else None
    y_val   = y[val_idx] if y is not None else None

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=64, shuffle=False
    )

    return train_loader, val_loader, X_train.shape[1], y_train, y_val, X_val


# ══════════════════════════════════════════════════════════════════════════════
# 2. 基础构建块
# ══════════════════════════════════════════════════════════════════════════════

class _Encoder(nn.Module):
    """隐变量编码器"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    """隐变量解码器（用于 AE/MAE）"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 自编码器 (AE)
# ══════════════════════════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    """全连接自编码器 — 重构归一化光谱"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """返回隐变量"""
        return self.encoder(x)


def train_autoencoder(train_loader, val_loader, input_dim, latent_dim,
                      epochs=200, lr=1e-3, patience=20):
    """
    训练自编码器。

    返回:
        model: 训练好的 Autoencoder（移到 cpu 上）
        history: dict, train_loss 和 val_loss 列表
    """
    model = Autoencoder(input_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)
            recon = model(x_batch)
            loss = F.mse_loss(recon, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                recon = model(x_batch)
                val_loss += F.mse_loss(recon, x_batch).item() * len(x_batch)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch == 1 or epoch % 20 == 0):
            print(f"  AE [{epoch:3d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} best={best_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  AE early stop @ epoch {epoch}, best val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    model.to('cpu')
    model.eval()
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 4. 掩码自编码器 (MAE)
# ══════════════════════════════════════════════════════════════════════════════

class MaskedAutoencoder(nn.Module):
    """
    掩码自编码器 — 随机遮罩 75% 波长点，重构被遮罩部分。
    """
    def __init__(self, input_dim, latent_dim, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, input_dim)

    def forward(self, x, mask=None):
        """返回 (reconstructed_full, mask)"""
        if mask is None:
            # 生成随机掩码
            batch_size, n_dims = x.shape
            n_mask = int(n_dims * self.mask_ratio)
            # 每个样本独立随机掩码
            mask = torch.zeros(batch_size, n_dims, device=x.device, dtype=torch.bool)
            for i in range(batch_size):
                perm = torch.randperm(n_dims, device=x.device)
                mask[i, perm[:n_mask]] = True
        # 将被遮罩位置置 0
        x_masked = x.clone()
        x_masked[mask] = 0.0
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, mask

    def encode(self, x):
        """返回隐变量（推理时无需掩码）"""
        return self.encoder(x)


def train_mae(train_loader, val_loader, input_dim, latent_dim,
              mask_ratio=0.75, epochs=200, lr=1e-3, patience=20):
    """
    训练掩码自编码器。

    损失: 仅计算被遮罩位置的 MSE。
    """
    model = MaskedAutoencoder(input_dim, latent_dim, mask_ratio).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)
            recon, mask = model(x_batch)
            # 仅计算掩码位置损失
            loss = F.mse_loss(recon[mask], x_batch[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                recon, mask = model(x_batch)
                val_loss += F.mse_loss(recon[mask], x_batch[mask]).item() * len(x_batch)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch == 1 or epoch % 20 == 0):
            print(f"  MAE [{epoch:3d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} best={best_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  MAE early stop @ epoch {epoch}, best val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    model.to('cpu')
    model.eval()
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. 对比学习 (SimCLR 风格)
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveEncoder(nn.Module):
    """
    对比学习编码器 — SimCLR 风格。

    训练时:
      对每条光谱做两次随机增广 → encoder → 投影头 → NT-Xent 损失
    推理时:
      去掉投影头，直接使用 encoder 的输出作为隐变量特征。

    增广策略（用于光谱）:
      - 高斯噪声
      - 幅度缩放
      - 波长偏移 / 基线偏移（可选）
    """
    def __init__(self, input_dim, latent_dim, projection_dim=64):
        super().__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        """训练时使用: encoder → projection head"""
        z = self.encoder(x)
        return self.projection(z)

    def encode(self, x):
        """推理时使用: 仅 encoder"""
        return self.encoder(x)


def _spectral_augment(x, noise_std=0.02, scale_range=(0.85, 1.15)):
    """
    光谱数据增广: 高斯噪声 + 幅度缩放。
    输入 x: (batch, n_dims)，输出与输入同形状。

    仅用于对比学习的正例对构建。
    """
    # 幅度缩放
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(*scale_range)
    x_aug = x * scale
    # 高斯噪声
    noise = torch.randn_like(x_aug) * noise_std
    return torch.clamp(x_aug + noise, 0.0, None)


def nt_xent_loss(z, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失。

    z: (2*batch, projection_dim) — 前半 batch 是 view₁, 后半是 view₂
    """
    batch_size = z.shape[0] // 2
    z = F.normalize(z, dim=1)

    # 相似度矩阵
    sim = torch.mm(z, z.t()) / temperature

    # 掩码: 排除自身
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim[~mask].view(2 * batch_size, -1)

    # 正例对: (i, i+batch_size) 和 (i+batch_size, i)
    # 移除对角后，正例索引修正
    # 行 i<B: 正例在 i+B (原) → i+B-1 (移除了对角 i)
    # 行 i>=B: 正例在 i-B (原) → i-B (不受影响)
    pos = torch.cat([
        torch.arange(batch_size - 1, 2 * batch_size - 1, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = F.cross_entropy(sim, pos)
    return loss


def train_contrastive(train_loader, val_loader, input_dim, latent_dim,
                      projection_dim=64, temperature=0.5,
                      epochs=200, lr=1e-3, patience=20):
    """
    训练对比学习编码器 (SimCLR)。

    返回:
        model: ContrastiveEncoder，已移到 cpu
        history: dict
    """
    model = ContrastiveEncoder(input_dim, latent_dim, projection_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)
            # 两个增广视图
            x1 = _spectral_augment(x_batch)
            x2 = _spectral_augment(x_batch)
            z1 = model(x1)
            z2 = model(x2)
            z = torch.cat([z1, z2], dim=0)
            loss = nt_xent_loss(z, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                x1 = _spectral_augment(x_batch)
                x2 = _spectral_augment(x_batch)
                z1 = model(x1)
                z2 = model(x2)
                z = torch.cat([z1, z2], dim=0)
                val_loss += nt_xent_loss(z, temperature).item() * len(x_batch)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch == 1 or epoch % 20 == 0):
            print(f"  Contrastive [{epoch:3d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Contrastive early stop @ epoch {epoch}, best val_loss={best_val_loss:.4f}")
            break

    model.load_state_dict(best_state)
    model.to('cpu')
    model.eval()
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. 特征提取接口
# ══════════════════════════════════════════════════════════════════════════════

def extract_latent_features(encoder, data):
    """
    用训练好的 encoder 提取隐变量特征。

    参数:
        encoder: nn.Module, 实现了 encode(x) → (n, latent_dim)
        data: dict, 包含光谱数据

    返回:
        latent: np.ndarray (n_spectra, latent_dim)
    """
    inorm_mat = compute_features(data).astype(np.float32)
    model = encoder.to(DEVICE)
    model.eval()
    with torch.no_grad():
        latent = model.encode(torch.from_numpy(inorm_mat).to(DEVICE))
    return latent.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 7. 统一训练入口
# ══════════════════════════════════════════════════════════════════════════════

PRETRAIN_METHODS = ['ae', 'mae', 'contrastive']

def train_pretrain(data_dicts, method, latent_dim, **kwargs):
    """
    统一入口: 训练指定方法的预训练模型。

    参数:
        data_dicts: list[dict], 各煤种 data dict
        method: 'ae' / 'mae' / 'contrastive'
        latent_dim: 隐变量维度
        **kwargs: 传递给具体训练函数的额外参数

    返回:
        encoder: nn.Module (CPU, eval mode)
        history: dict
    """
    train_loader, val_loader, input_dim, _, _, _ = build_pretrain_dataset(
        data_dicts, val_ratio=kwargs.pop('val_ratio', 0.15)
    )

    print(f"\n{'='*60}")
    print(f"训练 {method.upper()}  latent_dim={latent_dim}  input_dim={input_dim}")
    print(f"{'='*60}")

    if method == 'ae':
        return train_autoencoder(train_loader, val_loader, input_dim, latent_dim, **kwargs)
    elif method == 'mae':
        return train_mae(train_loader, val_loader, input_dim, latent_dim, **kwargs)
    elif method == 'contrastive':
        return train_contrastive(train_loader, val_loader, input_dim, latent_dim, **kwargs)
    else:
        raise ValueError(f"未知预训练方法: {method}，可选 {PRETRAIN_METHODS}")
