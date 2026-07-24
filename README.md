# LIBS 煤炭发热量预测 — Baseline 教程

**任务**: 通过激光诱导击穿光谱（LIBS）预测煤炭发热量（kcal/kg）  
**评估**: 线上 RMSE（越低越好）  
**当前版本**: V8  默认 Contrastive-32 + RidgeCV，CV-RMSE ≈ 184；可选 PCA（详见下方说明）

> 本项目基于 [Datawhale](https://datawhale.cn) 教学赛项目改造，赛题来源：[AI 实战训练营 — LIBS 煤炭发热量预测](https://ailc.datawhale.cn/hall/group/100000937)

---

## 项目结构

```
LIBS/
├── pyproject.toml     # uv 项目配置 & 依赖声明
├── uv.lock            # 锁定依赖版本
├── README.md          # 本文件
├── config.py          # 所有超参数 & 路径（改这里即可）
├── train.py           # 一键运行入口（5 步流程）
│
├── src/
│   ├── data.py                 # 数据加载（光谱读取 + 标签解析）
│   ├── features.py             # 特征工程（谱线积分 + 统计量 + PCA）
│   ├── model.py                # 两阶段 Ridge 训练 + 推理
│   ├── feature_extractors.py   # 特征提取器工厂 (PCA/Contrastive/AE/MAE)
│   ├── predictors.py           # 预测器统一接口 (RidgeCV/XGBoost/RF/GBR/MLP)
│   ├── pretrain.py             # 预训练编码器 (AE/MAE/Contrastive)
│   ├── pretrain_eval.py        # 预训练模型质量评测
│   ├── submit.py               # 打包 submit.zip
│   ├── experiment_tracker.py   # 跨轮次实验日志（记录 CV-RMSE）
│   └── window_search.py        # 窗口宽度自动搜索工具
│
├── train_data/        # 训练集（已解压）
├── test_data/         # 测试集（已解压）
├── submit_sample/     # 提交格式样例
└── output/            # 生成文件（experiment_log.csv / submit.csv / submit.zip）
```

---

## 快速开始

```bash
# 安装 uv（如未安装）
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 训练 + 推理 + 打包
python train.py

# 提交 output/submit.zip 到平台
```

---

## 方法说明

### 核心思路: 两阶段回归

```
LIBS 光谱 (7305维, 196-813nm)
        │
        ▼  Stage 1: 光谱 → 辅助指标
   全水分 / 灰分 / 氢 / 硫  (OOF 预测)
        │
        ▼  Stage 2: [光谱特征 + 辅助指标] → 发热量
   发热量 Q (kcal/kg)
```

**为什么两阶段?**  
辅助指标（灰分↑ → 发热量↓，氢↑ → 发热量↑）与目标强相关，
先预测辅助指标相当于把物理先验注入模型。

### 特征设计

| 特征 | 维度 | 关键思路 |
|---|---|---|
| 特征提取（默认 Contrastive-32） | 32 | 对比学习编码器，替代 PCA（config 可切 PCA） |
| 统计特征 | 17 | 偏度/峰度/熵反映谱线丰富程度 |
| 谱线积分 | 11×2 | C(247.86nm)/H(656.3nm)/灰分元素 |
| 物理比值 | 4 | 可燃/灰分、H/C 等 |

### 防过拟合策略

1. **正则化**: Ridge，alpha ∈ {1, 10, 50, 100, 500, 1000, 5000, 10000}（去掉 0.01/0.1；注意候选值过多会导致 alpha 选择本身过拟合）
2. **GroupKFold CV**: 以批次为单位分组，防止同批次光谱泄露
3. **均值收缩**: 批次数 ≤ 10 时，预测值向煤种均值收缩
4. **OOF**: Stage2 输入的辅助指标来自 Out-of-Fold 预测

### 全局 CV-RMSE 计算（v2026-07-15 更新）

采用**测试加权 pooled** 方式对齐线上评测：

1. **煤种内**: 汇集所有 OOF 批次预测值算 pooled RMSE（而非平均各 fold 的 RMSE）
2. **跨煤种**: 按各煤种**测试集批次数**平方加权再开方（而非按训练集批次数加权平均）

```
global = sqrt( Σ(rmse_i² × n_test_i) / Σ(n_test_i) )
```

这样每个测试批次等权，与线上评分 `sqrt(mean(true-pred)²)` 一致。
基线测试加权 CV-RMSE ≈ 188（线上 278），与线上分数变化方向协同性优于旧方法。

---

## 调参建议

修改 `config.py` 中的参数:

```python
# 特征提取方式: "contrastive_32" / "pca" / "ae_16" / "mae_16" 等
FEATURE_EXTRACTOR = "contrastive_32"

# PCA 最大保留维数（FEATURE_EXTRACTOR="pca" 时生效）
N_PCA_MAX = 30

# Ridge 正则化候选值
ALPHAS = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# 均值收缩阈值（批次数 ≤ 此值时生效）
SMALL_BATCH_THRESHOLD = 10
```

---

## 实验追踪

实验日志必须同时维护两个文件：

- **`output/experiment_log.csv`** — 结构化日志（程序自动追加），训练完成后手动填入 `test_score`
- **`EXPERIMENT_LOG.md`** — 可读日志（手动维护），每次有意义的实验追加新章节，包含目的、参数、分析

---

## 进一步优化方向

- [ ] 批次内光谱异常检测（去除污染光谱）
- [ ] 多版本集成（Ensemble V6 + V7 预测）

> **已排除的方向（经实验验证）：** （注：以下 CV-RMSE 采用当时的 per-fold 平均法计算，与当前测试加权 pooled 值不可直接比较，但线上分数和实验结论不变）
> - 树模型（LGBM/XGBoost）替换 RidgeCV — 线上 283.89 vs RidgeCV 278.50，小样本下非线性能力有害。详见实验 #5~#15。
> - 预测值异常值剔除后取中位数 — CV↓1.12 但线上↑6.3，CV乐观偏差。详见实验 #17。
> - **（现状：已作为默认方案）对比学习(Contrastive-32) 替换 PCA → 线上 241.86，↓36.64 (13.2%)** — AE/MAE/VAE/1D-CNN 四种方案全面退化（隐变量维度关联与 RidgeCV 独立贡献假设冲突），但对比学习的隐式正则化效果足以超越 PCA。默认 `python train.py` 即使用此方案。详见实验 2026-07-16。
> - **光谱级数据增强** — 6 种策略（批次内 Mixup/高斯噪声/幅值抖动/组合/折内跨批次 Mixup/物理散粒噪声）50+ 参数组合全部不优于基线。小样本定向 Mixup 虽获 CV 改善 164.98，但线上 290.29 证伪。RidgeCV 正则化已饱和，合成数据不带来新信息。详见实验 #26~#74。
> - **Boltzmann 图等离子体温度** — 新增 6 维特征（Fe I 谱线温度 + Ca II/Ca I、Hα/C、Si/Mg 比值），CV-RMSE 184.65→180.13↓4.52，但线上 241.86→243.99↑2.13。CV 改善不泛化到线上，与异常值剔除等方向相同的「CV 乐观偏差」模式。详见实验 #77。
