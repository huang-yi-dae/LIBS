"""
特征提取器工厂 — 统一管理 PCA / AE / MAE / Contrastive 等特征提取方式

用法:
    from src.feature_extractors import get_extractor
    encoder = get_extractor("contrastive_32")  # 返回编码器对象
    encoder = get_extractor("pca")             # 返回 None（使用 PCA）

所有预训练编码器的 .pt 文件置于 output/pretrained/ 目录下。
"""

import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_PRETRAINED_DIR = os.path.join(_PROJECT_ROOT, "output", "pretrained")

# ── 注册表 ─────────────────────────────────────────────────────────────────────
# name → {"type": "pca" / "encoder", "file": 文件名 (预训练路径) / None}
REGISTRY = {
    "pca":            {"type": "pca",     "file": None},
    "contrastive_8":  {"type": "encoder", "file": "contrastive_latent8.pt"},
    "contrastive_16": {"type": "encoder", "file": "contrastive_latent16.pt"},
    "contrastive_32": {"type": "encoder", "file": "contrastive_latent32.pt"},
    "ae_8":           {"type": "encoder", "file": "ae_latent8.pt"},
    "ae_16":          {"type": "encoder", "file": "ae_latent16.pt"},
    "ae_32":          {"type": "encoder", "file": "ae_latent32.pt"},
    "mae_8":          {"type": "encoder", "file": "mae_latent8.pt"},
    "mae_16":         {"type": "encoder", "file": "mae_latent16.pt"},
    "mae_32":         {"type": "encoder", "file": "mae_latent32.pt"},
}


def get_extractor(name):
    """
    获取特征提取器。

    参数:
        name: 注册表中的名称，如 "contrastive_32"、"pca"

    返回:
        encoder: nn.Module or None（None 表示使用 PCA）
    """
    if name not in REGISTRY:
        raise ValueError(
            f"未知特征提取器: {name}。可选: {', '.join(REGISTRY.keys())}"
        )

    info = REGISTRY[name]

    # PCA 路径：返回 None，主流程自动使用 build_feature_matrix
    if info["type"] == "pca":
        return None

    # 编码器路径：加载 .pt 文件
    pt_path = os.path.join(_PRETRAINED_DIR, info["file"])
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"编码器文件不存在: {pt_path}\n"
            f"请先运行 eval_pretrain.py --methods {name.split('_')[0]} "
            f"--latent-dims {name.split('_')[1]} 训练并保存模型。"
        )

    ckpt = torch.load(pt_path, map_location="cpu")
    input_dim = ckpt.get("input_dim", 7305)
    latent_dim = ckpt.get("latent_dim", 32)

    # 按名称前缀创建对应模型
    prefix = name.split("_")[0]
    if prefix == "contrastive":
        from src.pretrain import ContrastiveEncoder
        model = ContrastiveEncoder(input_dim, latent_dim)
    elif prefix == "ae":
        from src.pretrain import Autoencoder
        model = Autoencoder(input_dim, latent_dim)
    elif prefix == "mae":
        from src.pretrain import MaskedAutoencoder
        model = MaskedAutoencoder(input_dim, latent_dim)
    else:
        raise ValueError(f"未知编码器前缀: {prefix}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
