import argparse
import os
import random
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch_mimicry as mmc

from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.evaluate import evaluate_drs_with_index
from diagan.utils.settings import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--weight_path", default="./exp_results/cifar10/ldrd_weights.p", type=str)
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--netG_ckpt_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')
    parser.add_argument("--use_original_netD", action='store_true')
    parser.add_argument("--index_num", default=100, type=int, help="number of index to use for FID score")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    else:
        device = "cpu"
    
    # load model
    assert args.netG_ckpt_step
    print(f'load model from {save_path} step: {args.netG_ckpt_step}')
    netG, _, netD_drs, _, _, _ = get_gan_model(
        dataset_name=args.dataset, 
        model=args.model,
        reweight=False, 
        drs=True,
    )
    if not args.netG_train_mode:
        netG.eval()
        netD_drs.eval()
        netG.to(device)
        netD_drs.to(device)
    # step = netG.restore_checkpoint(ckpt_file=args.netG_ckpt_path)

    if args.dataset == 'celeba':
        dataset = 'celeba_64'
    else:
        dataset = args.dataset

    weight = pkl.load(open(args.weight_path, 'rb'))
    weight = np.array(weight)
    sort_index = np.argsort(weight)
    high_index = sort_index[-args.index_num:]
    low_index = sort_index[:args.index_num]
    weight_name = str(args.weight_path).split('/')[-1].split('.')[0]

    print(args)

    # Evaluate fid with index of high weight
    evaluate_drs_with_index(
        metric='fid',
        index=high_index,
        log_dir=save_path,
        netG=netG,
        netD_drs=netD_drs,
        dataset=dataset,
        num_fake_samples=50000,
        evaluate_step=args.netG_ckpt_step,
        num_runs=1,
        device=device,
        stats_file=None,
        use_original_netD=args.use_original_netD,
        name=f'high_{weight_name}',)

    # Evaluate fid with index of low weight
    evaluate_drs_with_index(
        metric='fid',
        index=low_index,
        log_dir=save_path,
        netG=netG,
        netD_drs=netD_drs,
        dataset=dataset,
        num_fake_samples=50000,
        evaluate_step=args.netG_ckpt_step,
        num_runs=1,
        device=device,
        stats_file=None,
        use_original_netD=args.use_original_netD,
        name=f'low_{weight_name}',)

if __name__ == '__main__':
    main()