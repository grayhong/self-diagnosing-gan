import argparse
import os
from pathlib import Path
import pickle
import numpy as np

from diagan.datasets.get_celeba_index_with_attr import get_celeba_index_with_attr
from diagan.utils.plot import calculate_scores


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="celeba", type=str)
    parser.add_argument("--root", "-r", default="./dataset/celeba", type=str, help="dataset dir")
    parser.add_argument("--attr", default="Bald", type=str, help="attribute name")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument('--p1_step', default=60000, type=int)
    parser.add_argument('--resample_score', type=str)
    opt = parser.parse_args()

    return opt

def main():
    args = parse_option()
    print(args)

    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)

    logit_path = save_path / 'logits_netD_eval.pkl'
    print(f'Use logit from: {logit_path}')
    logits = pickle.load(open(logit_path, "rb"))
    score_start_step = (args.p1_step - 5000)
    score_end_step = args.p1_step
    score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)
    sample_weights = score_dict[args.resample_score]
    print(f'sample_weights mean: {sample_weights.mean()}, var: {sample_weights.var()}, max: {sample_weights.max()}, min: {sample_weights.min()}')
    
    train_num = 162770
    attr_index, not_attr_index = get_celeba_index_with_attr(args.root, args.attr)
    attr_index = np.array(attr_index)
    not_attr_index = np.array(not_attr_index)
    attr_index = attr_index[attr_index < train_num]
    not_attr_index = not_attr_index[not_attr_index < train_num]
    attr_weights = sample_weights[attr_index]
    not_attr_weights = sample_weights[not_attr_index]
    print(f'attr weights mean: {attr_weights.mean()}')
    print(f'not attr weights mean: {not_attr_weights.mean()}')

if __name__ == '__main__':
    main()
