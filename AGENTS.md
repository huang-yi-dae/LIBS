# Repository Guidelines

## Project Structure & Module Organization

```
LIBS/
├── config.py              # All hyperparameters & paths (single point of control)
├── train.py               # Entry point: 5-step pipeline (load → train → log → predict → pack)
├── eval_pretrain.py       # Part 1 entry: pretrained-encoder quality eval (recon / latent-Y / linear-probe)
├── eval_combined.py       # Part 2 entry: (pretrain method × predictor) GroupKFold CV comparison
├── eval_proxy.py          # Offline proxy eval (proxy_lobo + adversarial AUC, online-consistency gate)
├── eval_combos.py         # Combo-direction offline eval (robust agg / shrink / aux-filter / cross-coal share)
├── pyproject.toml         # uv project manifest & dependency declarations
│
├── src/
│   ├── data.py            # Spectrum CSV loading + label Excel parsing
│   ├── features.py        # Feature engineering: spectral statistics, line integrals, PCA
│   ├── model.py           # Two-stage Ridge regression with GroupKFold CV + mean shrinkage
│   ├── feature_extractors.py  # Feature extractor factory (PCA/Contrastive/AE/MAE)
│   ├── predictors.py      # Unified predictor interface (RidgeCV/XGBoost/RF/GBR/MLP)
│   ├── pretrain.py        # Pretrained encoders (AE/MAE/Contrastive)
│   ├── pretrain_eval.py   # Pretrained model quality evaluation (helper used by eval_pretrain.py)
│   ├── submit.py          # Package predictions into submit.zip
│   ├── experiment_tracker.py  # Cross-run experiment logging (CSV-based)
│   └── window_search.py   # Automated window width search for KEY_LINES
│
├── train_data/            # Training spectra (organized by coal type → batch folders)
├── test_data/             # Test spectra (same layout as train_data)
├── submit_sample/         # Submission format template
└── output/                # Generated artifacts: experiment_log.csv, submit.csv, submit.zip
```

Source code lives entirely under `src/`. Each module has a single, documented responsibility. Data directories are flat by coal type, with batch folders nested inside.

## Build, Test, and Development Commands

```bash
uv sync              # Install all dependencies from uv.lock (use after clone or dependency change)
python train.py      # Run the full pipeline: training → evaluation → test inference → packaging
python eval_pretrain.py --quick                                      # Part 1: 预训练模型质量评测（快速模式）
python eval_combined.py --pretrained contrastive --mode two-stage    # Part 2: 组合评测（对比学习+全部预测器）
```

> Note: the spectral-augmentation and spectrum-perturbation CLI flags (`--augment`, `--fold-mixup`, `--wave-shift`, `--baseline-*`) were removed from `train.py` — those experiment directions were excluded by prior experiments. Only the plain `python train.py` pipeline remains. `src/augment.py` and `src/perturb.py` no longer exist.

There are no separate build or test scripts. Validation is done locally via CV-RMSE reported by `train.py`, and online by submitting `output/submit.zip` to the competition platform.

## Coding Style & Naming Conventions

- **Formatter**: None enforced. Prefer the standard library and numpy/scikit-learn idioms.
- **Docstrings**: Every module and function has a PEP 257-style docstring explaining its responsibility, parameters, and return values.
- **Types**: Use Python type hints for public function signatures (e.g., `-> None`, `-> dict`).
- **Naming**: Functions and variables use `snake_case`. Module-level constants use `UPPER_CASE`. Private helpers prefixed with `_`.
- **Imports**: Group as stdlib → third-party → local modules, separated by blank lines.
- **Line length**: Aim for ≤ 100 characters. Inline comments explain *why*, not *what*.

## Testing Guidelines

This project currently has **no automated test suite**. Testing is manual:

- Run `python train.py` end-to-end and verify no errors.
- Check `output/experiment_log.csv` for the new CV-RMSE entry.
- Visually inspect `output/submit.csv` for sensible predictions (e.g., no NaN, within expect
ed kcal/kg range).

If you add tests, place them in a `tests/` directory mirroring `src/`, name files `test_*.py`, and document the test framework in this section.

## Commit & Pull Request Guidelines

- **Commit messages**: Use the `<type>: <description>` format, e.g., `feat: add spectrum anomaly detection` or `fix: handle empty batch directory in load_coal_spectra`. Keep the first line under 72 characters.
- **Branch naming**: Use `feat/`, `fix/`, or `chore/` prefixes, e.g., `feat/xgb-stage2`.
- **Pull requests**: Include a short summary of the change, the motivation (e.g., "Reduces CV-RMSE by 5 points"), and any relevant CV-RMSE before/after numbers. Link to tracked issues if applicable.
- **Single commit per logical change** is preferred; squash before merging.

## Experiment Tracking

实验日志必须同时记录到以下两个文件：

- **`output/experiment_log.csv`** — 结构化实验记录（程序自动追加）。每次运行 `train.py` 自动写入一条 `timestamp,cv_rmse,test_score,treatment` 记录。线上提交后手动填写 `test_score`。
- **`EXPERIMENT_LOG.md`** — 可读实验记录（手动维护）。每次有意义的实验（尤其是线上提交后），必须在此文件追加新章节，包含实验目的、参数、CV-RMSE、线上得分、分析结论。

两条记录必须对应同一实验，`treatment` 描述保持一致。

## 实验记录写作规范 (Style Guide)

双文件实验记录必须遵循以下统一格式，保证可读性与可检索性。

### experiment_log.csv（机器可读，程序追加）
- 固定 4 列 `timestamp,cv_rmse,test_score,treatment`；**纯 CSV，文件内禁止写 `#` 注释**（变更说明写在 EXPERIMENT_LOG.md）。
- `timestamp`：`log_experiment` 自动写入的真实运行时间 `YYYY-MM-DD HH:MM:SS`，不得用占位/批量假时间戳。
- `treatment`：简洁中文，约定如下：
  - 模型名用中文或约定缩写：对比学习 / RidgeCV / LGBM / XGBoost / PCA / AE / MAE / VAE / 1D-CNN。
  - 变体维度写为 `对比学习-32`、`AE-16`（连字符 + 数字）。
  - 两阶段组合写为 `Stage1 X, Stage2 Y`；两阶段同模型写为 `两阶段 X`。
  - 不写句末句号；补充说明用全角括号，如 `（测试加权 pooled）`、`（线上验证）`、`（最佳）`。
  - 未填写处理时统一记 `（未记录处理）`，不得留空。
  - 示例：`对比学习-32 + RidgeCV（两阶段）`、`Stage1 LGBM, Stage2 RidgeCV`、`KEY_LINES 谱线修正`。

### EXPERIMENT_LOG.md（人工维护）
- 每个实验用 `## 实验 N — 标题`，**N 全局唯一、顺序递增**，不得与已有编号（含 `6~15` 等区间组）冲突，不得用纯字母编号。
- 方法论/流程变更类说明用 `## 方法论 — 日期: 标题`，与单实验章节区分。
- `时间` 字段统一为 `YYYY-MM-DD HH:MM[:SS]`；纯日期保留 `YYYY-MM-DD`；时间区间用 `YYYY-MM-DD HH:MM ~ HH:MM`（全角 `~`）。
- 表格字段建议统一：时间 / 特征 / CV-RMSE / 线上得分 / 处理 / 结论。
- 文内引用其他实验用 `实验 #N`，N 必须与标题编号一致。
- “已排除方向汇总”表的 `#` 列为方向序号（1~13），与实验编号相互独立，勿混用。

## Evaluation Metric (CV-RMSE)

CV-RMSE is the offline proxy for the online score and is how every experiment is compared.

- **Per coal type**: pool all out-of-fold (OOF) predictions within the type and compute one
  pooled RMSE (NOT the average of per-fold RMSEs).
- **Global**: weighted by each coal type's *test-set* batch count, so every test batch is
  equally weighted — this matches the online score `sqrt(mean((true − pred)²))`:

  ```
  global = sqrt( Σ(rmse_i² × n_test_i) / Σ(n_test_i) )
  ```

  where `rmse_i` is coal type i's pooled OOF RMSE and `n_test_i` is its test-batch count.
- Per-fold averaging is no longer used. Always read a CV-RMSE under this definition.

## 实验对比准则

每次实验的 CV-RMSE 和线上得分必须同时与**当前最优版本**对比，而非仅与初始基线对比。当前最优版本记录在 project memory 中（如 Contrastive-32: CV≈184, 线上 241.86）。

红线（正则化）: Ridge `ALPHAS` 不得加入 < 1.0 的值——α<1 在训练/测试分布偏移下必过拟合（线上单调劣化：α≈1.0→241.86, 0.3→248.33, 1e-3→255.62，见实验 #79/#80/#81 与 README §防过拟合策略）。以 CV-RMSE 下降为目标的 α 调参必须用线上验证做最终判据，不可仅凭 CV 增益采用。

对比格式示例：
- CV-RMSE: **xxx**（较最优 ↓/↑ xx）
- 线上得分：**xxx**（较最优 ↓/↑ xx）

当最优版本更新时，同步更新 AGENTS.md 中的基准值。
