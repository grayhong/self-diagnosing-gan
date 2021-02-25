import argparse
import pickle
from pathlib import Path

import numpy as np

from diagan.datasets.predefined import get_predefined_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="color_mnist", type=str)
    parser.add_argument("--root", "-r", default="./dataset/colour_mnist", type=str, help="dataset dir")
    parser.add_argument("--baseline_exp_path", default="color_mnist", type=str)
    parser.add_argument("--resample_exp_path", default="color_mnist", type=str)
    parser.add_argument('--p2_step', default=20000, type=int)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    args = parser.parse_args()

    baseline_exp_path = Path(args.baseline_exp_path)
    resample_exp_path = Path(args.resample_exp_path)

    baseline_ae_loss = np.load(baseline_exp_path / f'cae_checkpoints/{args.p2_step}_steps_seed1/cae_training_loss.npy')
    resample_ae_loss = np.load(resample_exp_path / f'cae_checkpoints/{args.p2_step}_steps_seed1/cae_training_loss.npy')
    baseline_ae = baseline_ae_loss[:, -1]
    resample_ae = resample_ae_loss[:, -1]

    ds_train = get_predefined_dataset(
        dataset_name=args.dataset,
        root=args.root,
        weights=None,
        major_ratio=args.major_ratio,
        num_data=args.num_data
    )

    idx_name = 'green'
    index = np.where(ds_train.dataset.biased_targets == 1)
    baseline_mean = baseline_ae[index].mean()
    resample_mean = resample_ae[index].mean()
    baseline_resample_diff = (resample_mean - baseline_mean) / baseline_mean * 100
    print(f'{idx_name}, baseline_mean: {baseline_mean}, resample_mean: {resample_mean} diff: {baseline_resample_diff}%')

    
if __name__ == '__main__':
    main()