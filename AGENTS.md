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

Every run automatically appends to `output/experiment_log.csv`. After submitting to the platform, manually fill in `test_score` and `treatment` (a short description of the change) to maintain a reproducible log of iterations.
