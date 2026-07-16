# Repository Guidelines

## Project Structure & Module Organization

```
LIBS/
├── config.py              # All hyperparameters & paths (single point of control)
├── train.py               # Entry point: 5-step pipeline (load → train → log → predict → pack)
├── pyproject.toml         # uv project manifest & dependency declarations
│
├── src/
│   ├── data.py            # Spectrum CSV loading + label Excel parsing
│   ├── features.py        # Feature engineering: spectral statistics, line integrals, PCA
│   ├── model.py           # Two-stage Ridge regression with GroupKFold CV + mean shrinkage
│   ├── feature_extractors.py  # Feature extractor factory (PCA/Contrastive/AE/MAE)
│   ├── predictors.py      # Unified predictor interface (RidgeCV/XGBoost/RF/GBR/MLP)
│   ├── pretrain.py        # Pretrained encoders (AE/MAE/Contrastive)
│   ├── pretrain_eval.py   # Pretrained model quality evaluation
│   ├── submit.py          # Package predictions into submit.zip
│   ├── augment.py         # Spectral augmentation (shot-noise/mixup/jitter) — rejected
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
python train.py --augment shot-noise --aug-noise-factor 0.05 --aug-factor 2  # 物理散粒噪声增强（实验已排除方向）
python train.py --fold-mixup --fold-mixup-alpha 1.0 --fold-mixup-factor 1    # 折内跨批次 Mixup（实验已排除方向）
python eval_pretrain.py --quick                                      # Part 1: 预训练模型质量评测（快速模式）
python eval_combined.py --pretrained contrastive --mode two-stage    # Part 2: 组合评测（对比学习+全部预测器）
```

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
