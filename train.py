"""
主入口 — 运行方式: python train.py

流程:
  1. 加载标签
  2. 按煤种分别训练两阶段模型
  3. 记录实验日志（timestamp + CV-RMSE）
  4. 测试集推理
  5. 打包 submit.zip
"""

import sys
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 确保 src/ 包可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR, FEATURE_EXTRACTOR
from src.feature_extractors import get_extractor
from src.data     import load_labels, load_coal_spectra
from src.model    import train_coal_model, predict_coal
from src.submit   import pack_submission
from src.experiment_tracker import log_experiment
from src.augment  import augment_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", type=str, default="",
                        help="实验处理描述，写入 experiment_log.csv")
    parser.add_argument("--augment", type=str, default="none",
                        choices=["none", "noise", "shot-noise", "mixup", "jitter", "combined"],
                        help="数据增强策略: none=不增强")
    parser.add_argument("--aug-factor", type=int, default=1,
                        help="每条原始光谱生成的增强副本数")
    parser.add_argument("--aug-alpha", type=float, default=None,
                        help="Mixup Beta α 参数 (默认0.5) / Noise factor / Jitter范围")
    parser.add_argument("--aug-noise-factor", type=float, default=None,
                        help="噪声强度 (默认 0.02)")
    parser.add_argument("--aug-jitter-min", type=float, default=None,
                        help="Jitter 缩放下限 (默认 0.9)")
    parser.add_argument("--aug-jitter-max", type=float, default=None,
                        help="Jitter 缩放上限 (默认 1.1)")
    parser.add_argument("--aug-correlation-length", type=float, default=None,
                        help="shot-noise 波长相关长度 (波长点数, 默认 3.0)")
    parser.add_argument("--aug-small-only", action="store_true",
                        help="仅对 SMALL_BATCH_THRESHOLD 以内的小样本煤种做增强")
    # ── 光谱扰动（波长偏移 + 基线偏移）──
    parser.add_argument("--wave-shift", type=float, default=0.0,
                        help="波长偏移量(nm)，正值=红移(长波)，负值=蓝移(短波)")
    parser.add_argument("--baseline-type", type=str, default="none",
                        choices=["none", "constant", "linear", "polynomial", "sine"],
                        help="基线偏移类型")
    parser.add_argument("--baseline-amp", type=float, default=None,
                        help="constant/sine 基线的振幅")
    parser.add_argument("--baseline-slope", type=float, default=None,
                        help="linear 基线的斜率")
    parser.add_argument("--baseline-intercept", type=float, default=None,
                        help="linear 基线的截距")
    parser.add_argument("--baseline-freq", type=float, default=None,
                        help="sine 基线的频率")
    parser.add_argument("--baseline-phase", type=float, default=None,
                        help="sine 基线的相位")
    parser.add_argument("--baseline-poly-coeffs", type=str, default=None,
                        help="polynomial 基线系数，逗号分隔，如 '0,0.01,0.0001' 对应 c0 + c1*x + c2*x^2")
    parser.add_argument("--fold-mixup", action="store_true",
                        help="折内跨批次 Mixup: CV每折训练集内做特征级插值")
    parser.add_argument("--fold-mixup-alpha", type=float, default=1.0,
                        help="折内 Mixup Beta α 参数 (默认1.0, 即Uniform)")
    parser.add_argument("--fold-mixup-factor", type=int, default=1,
                        help="折内 Mixup 每条样本生成的混合副本数")
    args = parser.parse_args()

    # ── 构建扰动配置 ──
    perturb_cfg = None
    if abs(args.wave_shift) > 1e-6 or args.baseline_type != "none":
        baseline_kw = {}
        if args.baseline_amp is not None:
            baseline_kw["amp"] = args.baseline_amp
        if args.baseline_slope is not None:
            baseline_kw["slope"] = args.baseline_slope
        if args.baseline_intercept is not None:
            baseline_kw["intercept"] = args.baseline_intercept
        if args.baseline_freq is not None:
            baseline_kw["freq"] = args.baseline_freq
        if args.baseline_phase is not None:
            baseline_kw["phase"] = args.baseline_phase
        if args.baseline_poly_coeffs is not None:
            coeffs = [float(c) for c in args.baseline_poly_coeffs.split(",")]
            baseline_kw["coeffs"] = coeffs

        perturb_cfg = {
            "wave_shift": args.wave_shift,
            "baseline_type": args.baseline_type,
            "baseline_kw": baseline_kw if baseline_kw else None,
        }
        print(f"  扰动配置: {perturb_cfg}")

    # 自动拼接 treatment（如果没提供自定义描述）
    if not args.treatment and perturb_cfg is not None:
        parts = []
        if abs(args.wave_shift) > 1e-6:
            parts.append(f"波长偏移{args.wave_shift:+.1f}nm")
        if args.baseline_type != "none":
            bparts = [f"基线({args.baseline_type}"]
            if args.baseline_amp is not None:
                bparts.append(f"amp={args.baseline_amp}")
            if args.baseline_slope is not None:
                bparts.append(f"slope={args.baseline_slope}")
            if args.baseline_freq is not None:
                bparts.append(f"freq={args.baseline_freq}")
            parts.append(",".join(bparts) + ")")
        args.treatment = " ".join(parts)

    # ── 加载特征提取器 ──────────────────────────────────────────────────────
    encoder = get_extractor(FEATURE_EXTRACTOR)
    if encoder is not None:
        print(f"  特征提取: {FEATURE_EXTRACTOR}")
        if not args.treatment:
            args.treatment = FEATURE_EXTRACTOR
    else:
        print(f"  特征提取: PCA")

    # ── Step 1: 加载标签 ──────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 加载标签")
    label_map, aux_map = load_labels()
    print(f"  标签总数: {len(label_map)}")

    # ── Step 2: 按煤种训练 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 两阶段训练（分煤种）")

    # 预加载测试集各煤种批次数（作为全局 pooled CV-RMSE 的权重）
    test_n_batches = {}
    for ct in COAL_TYPES:
        td = load_coal_spectra(TEST_DIR, ct)
        test_n_batches[ct] = td['n_batches'] if td else 0
    total_test_batches = sum(test_n_batches.values())
    print(f"  测试集总批次: {total_test_batches} ({test_n_batches})")

    models      = {}
    cv_results  = {}
    test_w_l    = []  # 各煤种测试批次数，用于加权

    for coal_type in COAL_TYPES:
        train_data = load_coal_spectra(TRAIN_DIR, coal_type, label_map, aux_map)
        if train_data is None or train_data['n_batches'] == 0:
            print(f"\n  [{coal_type}] 未找到训练数据，跳过")
            continue

        # ── 数据增强（仅在训练时，保持 GroupKFold 结构） ──
        do_augment = (args.augment != "none")
        if do_augment and args.aug_small_only:
            # 仅小样本煤种做增强
            from config import SMALL_BATCH_THRESHOLD
            do_augment = (train_data['n_batches'] <= SMALL_BATCH_THRESHOLD)
            if not do_augment:
                print(f"  [{coal_type}] 跳过增强 ({train_data['n_batches']} batches > 阈值)")

        if do_augment:
            aug_kw = {'strategy': args.augment, 'aug_factor': args.aug_factor}
            if args.aug_alpha is not None:
                aug_kw['alpha'] = args.aug_alpha
            if args.aug_noise_factor is not None:
                aug_kw['noise_factor'] = args.aug_noise_factor
            if args.aug_jitter_min is not None:
                aug_kw['jitter_min'] = args.aug_jitter_min
            if args.aug_jitter_max is not None:
                aug_kw['jitter_max'] = args.aug_jitter_max
            if args.aug_correlation_length is not None:
                aug_kw['correlation_length'] = args.aug_correlation_length
            n_orig = len(train_data['spectra'])
            train_data = augment_data(train_data, **aug_kw)
            n_aug = len(train_data['spectra']) - n_orig
            print(f"  [{coal_type}] 增强: {n_orig} → {len(train_data['spectra'])} "
                  f"(+{n_aug} augmented, groups={train_data['n_batches']})")

        # ── 折内 Mixup 配置 ──
        fm_config = None
        if args.fold_mixup:
            fm_config = {
                'alpha': args.fold_mixup_alpha,
                'aug_factor': args.fold_mixup_factor,
            }

        model_dict = train_coal_model(coal_type, train_data,
                                          fold_mixup_config=fm_config,
                                          perturb_cfg=perturb_cfg,
                                          encoder=encoder)
        models[coal_type]      = model_dict
        cv_results[coal_type]  = model_dict['cv_rmse']
        test_w_l.append(test_n_batches[coal_type])

    # 全局 pooled RMSE: 各煤种 pooled RMSE 按测试集批次数平方加权
    # global = sqrt( Σ(rmse_i² × n_test_i) / Σ(n_test_i) )
    # 使得 CV-RMSE 的加权方式与线上评测一致（测试集各批次等权）
    test_w = np.array(test_w_l, dtype=np.float32)
    rmses  = np.array(list(cv_results.values()), dtype=np.float32)
    global_cv_rmse = float(np.sqrt(np.average(rmses ** 2, weights=test_w)))
    print(f"\n{'=' * 60}")
    print(f"全局 CV-RMSE: {global_cv_rmse:.2f}")

    # ── Step 3: 记录实验日志 ──────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 记录实验日志")
    log_experiment(global_cv_rmse, treatment=args.treatment)

    # ── Step 4: 测试集推理 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: 测试集推理")

    all_preds = {}
    for coal_type in COAL_TYPES:
        if coal_type not in models:
            continue
        test_data = load_coal_spectra(TEST_DIR, coal_type, label_map=None, aux_map=None)
        if test_data is None or len(test_data['spectra']) == 0:
            print(f"  [{coal_type}] 无测试数据")
            continue
        bp = predict_coal(coal_type, test_data, models[coal_type])
        all_preds.update(bp)

    print("\n  预测结果:")
    for name, pred in sorted(all_preds.items()):
        print(f"    {name}: {pred:.2f}")

    # ── Step 5: 打包提交 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: 生成提交文件")
    pack_submission(all_preds, cv_results, global_cv_rmse)
    print("\n完成！提交文件在 output/submit.zip")


if __name__ == "__main__":
    main()
