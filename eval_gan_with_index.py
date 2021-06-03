import argparse
import os
import random
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch_mimicry as mmc

from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.evaluate import evaluate_with_index
from diagan.utils.settings import set_seed
from diagan.utils.plot import calculate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument("--baseline_exp_name", type=str, help="exp name")
    parser.add_argument('--p1_step', default=40000, type=int)
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--netG_ckpt_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')
    parser.add_argument('--resample_score', type=str)
    parser.add_argument('--gold', action='store_true')
    parser.add_argument('--topk', action='store_true')
    parser.add_argument("--index_num", default=100, type=int, help="number of index to use for FID score")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    baseline_output_dir = f'{args.work_dir}/{args.baseline_exp_name}'
    baseline_save_path = Path(baseline_output_dir)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    else:
        device = "cpu"

    # load model
    assert args.netG_ckpt_step
    print(f'load model from {save_path} step: {args.netG_ckpt_step}')
    netG, _, _, _ = get_gan_model(
        dataset_name=args.dataset,
        model=args.model,
        loss_type=args.loss_type,
        topk=args.topk,
        gold=args.gold,
    )
    netG.to(device)
    if not args.netG_train_mode:
        netG.eval()

    if args.dataset == 'celeba':
        dataset = 'celeba_64'
        window = 5000
    else:
        dataset = args.dataset
        window = 5000

    logit_path = baseline_save_path / 'logits_netD_eval.pkl'
    print(f'Use logit from: {logit_path}')
    logits = pickle.load(open(logit_path, "rb"))
    score_start_step = (args.p1_step - window)
    score_end_step = args.p1_step
    score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)
    sample_weights = score_dict[args.resample_score]
    print(
        f'sample_weights mean: {sample_weights.mean()}, var: {sample_weights.var()}, max: {sample_weights.max()}, min: {sample_weights.min()}')

    print(args)

    sort_index = np.argsort(sample_weights)
    high_index = sort_index[-args.index_num:]
    low_index = sort_index[:args.index_num]

    # Evaluate fid with index of high weight
    evaluate_with_index(
        metric='fid',
        index=high_index,
        log_dir=save_path,
        netG=netG,
        dataset=dataset,
        num_fake_samples=50000,
        evaluate_step=args.netG_ckpt_step,
        num_runs=1,
        device=device,
        stats_file=None,
        name=f'high_{args.resample_score}', )

    # Evaluate fid with index of low weight
    evaluate_with_index(
        metric='fid',
        index=low_index,
        log_dir=save_path,
        netG=netG,
        dataset=dataset,
        num_fake_samples=50000,
        evaluate_step=args.netG_ckpt_step,
        num_runs=1,
        device=device,
        stats_file=None,
        name=f'low_{args.resample_score}', )


if __name__ == '__main__':
    main()