# LIBS 煤炭发热量预测 — Baseline 教程

**任务**: 通过激光诱导击穿光谱（LIBS）预测煤炭发热量（kcal/kg）  
**评估**: 线上 RMSE（越低越好）  
**当前版本**: V7  本地 CV-RMSE ≈ 176

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
│   ├── submit.py               # 打包 submit.zip
│   └── experiment_tracker.py   # 跨轮次实验日志（记录 CV-RMSE）
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
| PCA 降维光谱 | ≤30 | 整体光谱形状 |
| 统计特征 | 17 | 偏度/峰度/熵反映谱线丰富程度 |
| 谱线积分 | 11×2 | C(247.86nm)/H(656.3nm)/灰分元素 |
| 物理比值 | 4 | 可燃/灰分、H/C 等 |

### 防过拟合策略

1. **正则化**: Ridge，alpha ∈ {1, 10, 50, 100, 500, 1000, 5000, 10000}（去掉 0.01/0.1）
2. **GroupKFold CV**: 以批次为单位分组，防止同批次光谱泄露
3. **均值收缩**: 批次数 ≤ 10 时，预测值向煤种均值收缩
4. **OOF**: Stage2 输入的辅助指标来自 Out-of-Fold 预测

---

## 调参建议

修改 `config.py` 中的参数:

```python
# 增大 N_PCA_MAX 可捕获更多光谱变化（过多会过拟合）
N_PCA_MAX = 30

# 扩大 ALPHAS 范围（加入更大值）可加强正则化
ALPHAS = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# 调整收缩阈值
SMALL_BATCH_THRESHOLD = 10
```

---

## 实验追踪

每轮训练结果自动记录到 `output/experiment_log.csv`:

| timestamp | cv_rmse | test_score | treatment |
|-----------|---------|------------|-----------|
| 自动填入 | ✓ | 待填写 | 待填写 |

训练完成后手动填入 `test_score`（线上得分）和 `treatment`（本轮改动描述），方便跨轮次对比。

---

## 进一步优化方向

- [ ] LightGBM 替换 Stage2 Ridge（捕获非线性）
- [ ] 批次内光谱异常检测（去除污染光谱）
- [ ] Boltzmann 图估算等离子体温度（更深物理特征）
- [ ] 多版本集成（Ensemble V6 + V7 预测）
