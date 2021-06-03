import argparse
import os
import pickle
from pathlib import Path
import csv

import numpy as np

from diagan.datasets.predefined import get_predefined_dataset
from diagan.utils.plot import calculate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="color_mnist", type=str)
    parser.add_argument("--root", "-r", default="./dataset/colour_mnist", type=str, help="dataset dir")
    parser.add_argument("--baseline_exp_path", default="color_mnist", type=str)
    parser.add_argument("--resample_exp_path", default="color_mnist", type=str)
    parser.add_argument('--p1_step', default=15000, type=int)
    parser.add_argument('--p2_step', default=20000, type=int)
    parser.add_argument('--resample_score', type=str)
    parser.add_argument("--use_loss", action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    baseline_exp_path = Path(args.baseline_exp_path)
    resample_exp_path = Path(args.resample_exp_path)

    if args.use_loss:
        baseline_ae_loss = np.load(baseline_exp_path / f'cae_checkpoints/{args.p2_step}_steps_seed{args.seed}/cae_training_loss.npy')
        resample_ae_loss = np.load(resample_exp_path / f'cae_checkpoints/{args.p2_step}_steps_seed{args.seed}/cae_training_loss.npy')
        baseline_ae = baseline_ae_loss[:, -1]
        resample_ae = resample_ae_loss[:, -1]

    logits = pickle.load(open(baseline_exp_path / 'logits_netD_eval.pkl', "rb"))
    score_start_step = args.p1_step - 5000
    score_end_step = args.p1_step
    score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)
    sample_weights = score_dict[args.resample_score]
    
    weight_sort_index = np.argsort(sample_weights)
    test_dict = dict()


    ds_train = get_predefined_dataset(
        dataset_name=args.dataset,
        root=args.root,
        weights=None,
        major_ratio=args.major_ratio,
        num_data=args.num_data
    )

    csv_file = f'./re_{args.dataset}_{args.name}.csv'
    if os.path.exists(csv_file):
        f = open(csv_file, 'a', newline='')
        wr = csv.writer(f)
    else:
        f = open(csv_file, 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['Ratio', 'Seed', 'Type', 'Baseline', 'Resample', 'Difference(%)'])

    test_dict['all'] = weight_sort_index
    if args.dataset == 'color_mnist':
        test_dict['green'] = np.where(ds_train.dataset.biased_targets == 1)
    elif args.dataset == 'mnist_fmnist':
        test_dict['fmnist'] = np.where(ds_train.dataset.mixed_targets == 1)

    for idx_name, index in test_dict.items():
        baseline_mean = baseline_ae[index].mean()
        resample_mean = resample_ae[index].mean()
        baseline_resample_diff = (resample_mean - baseline_mean) / baseline_mean * 100
        print(f'{idx_name}, baseline_mean: {baseline_mean}, resample_mean: {resample_mean} diff: {baseline_resample_diff}%')
        wr.writerow([args.major_ratio, args.seed, idx_name, baseline_mean, resample_mean, baseline_resample_diff])

    f.close()

if __name__ == '__main__':
    main()