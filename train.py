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

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR
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
    parser.add_argument("--fold-mixup", action="store_true",
                        help="折内跨批次 Mixup: CV每折训练集内做特征级插值")
    parser.add_argument("--fold-mixup-alpha", type=float, default=1.0,
                        help="折内 Mixup Beta α 参数 (默认1.0, 即Uniform)")
    parser.add_argument("--fold-mixup-factor", type=int, default=1,
                        help="折内 Mixup 每条样本生成的混合副本数")
    args = parser.parse_args()
    # ── Step 1: 加载标签 ──────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 加载标签")
    label_map, aux_map = load_labels()
    print(f"  标签总数: {len(label_map)}")

    # ── Step 2: 按煤种训练 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 两阶段训练（分煤种）")

    models      = {}
    cv_results  = {}
    n_batches_l = []  # 各煤种批次数，用于加权平均

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

        model_dict = train_coal_model(coal_type, train_data, fold_mixup_config=fm_config)
        models[coal_type]      = model_dict
        cv_results[coal_type]  = model_dict['cv_rmse']
        n_batches_l.append(train_data['n_batches'])

    # 加权平均（按批次数加权，更多批次 = 更可靠的 CV 估计）
    weights = np.array(n_batches_l, dtype=np.float32)
    values  = np.array(list(cv_results.values()), dtype=np.float32)
    global_cv_rmse = float(np.average(values, weights=weights))
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
