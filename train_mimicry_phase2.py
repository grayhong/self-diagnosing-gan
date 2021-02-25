import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.trainer import LogTrainer
from diagan.utils.plot import (
    calculate_scores, print_num_params,
                                    show_sorted_score_samples
)
from diagan.utils.settings import set_seed


def get_dataloader(dataset, batch_size=128, weights=None, eps=1e-6):
    if weights is not None:
        weight_list = [eps if i < eps else i for i in weights]
        sampler = data.WeightedRandomSampler(weight_list, len(weight_list), replacement=True)
    else:
        sampler = None
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False if sampler else True,
        sampler=sampler,
        num_workers=8,
        pin_memory=True)
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--root", "-r", default="./dataset/cifar10", type=str, help="dataset dir")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--baseline_exp_name", type=str, help="exp name")
    parser.add_argument('--p1_step', default=40000, type=int)
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_steps', default=80000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--decay', default='linear', type=str)
    parser.add_argument('--n_dis', default=5, type=int)
    parser.add_argument('--resample_score', type=str)
    parser.add_argument('--gold', action='store_true')
    parser.add_argument('--topk', action='store_true')
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

    
    prefix = args.exp_name.split('/')[-1]

    if args.dataset == 'celeba':
        window = 5000
    elif args.dataset == 'cifar10':
        window = 5000
    else:
        window = 5000


    if not args.gold:
        logit_path = baseline_save_path / 'logits_netD_eval.pkl'
        print(f'Use logit from: {logit_path}')
        logits = pickle.load(open(logit_path, "rb"))
        score_start_step = (args.p1_step - window)
        score_end_step = args.p1_step
        score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)
        sample_weights = score_dict[args.resample_score]
        print(f'sample_weights mean: {sample_weights.mean()}, var: {sample_weights.var()}, max: {sample_weights.max()}, min: {sample_weights.min()}')
    else:
        sample_weights = None

    netG_ckpt_path = baseline_save_path / f'checkpoints/netG/netG_{args.p1_step}_steps.pth'
    netD_ckpt_path = baseline_save_path / f'checkpoints/netD/netD_{args.p1_step}_steps.pth'

    netD_drs_ckpt_path = baseline_save_path / f'checkpoints/netD/netD_{args.p1_step}_steps.pth'
    netG, netD, netD_drs, optG, optD, optD_drs = get_gan_model(
        dataset_name=args.dataset,
        model=args.model,
        loss_type=args.loss_type,
        drs=True,
        topk=args.topk,
        gold=args.gold,
    )
    
    print(f'model: {args.model} - netD_drs_ckpt_path: {netD_drs_ckpt_path}')
    

    print_num_params(netG, netD)

    ds_train = get_predefined_dataset(dataset_name=args.dataset, root=args.root, weights=None)
    dl_train = get_dataloader(ds_train, batch_size=args.batch_size, weights=sample_weights)

    ds_drs = get_predefined_dataset(dataset_name=args.dataset, root=args.root, weights=None)
    dl_drs = get_dataloader(ds_drs, batch_size=args.batch_size, weights=None)

    if not args.gold:
        show_sorted_score_samples(ds_train, score=sample_weights, save_path=save_path, score_name=args.resample_score, plot_name=prefix)
    
    print(args)

    # Start training
    trainer = LogTrainer(
        output_path=save_path,
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        netG_ckpt_file=str(netG_ckpt_path),
        netD_ckpt_file=str(netD_ckpt_path),
        netD_drs_ckpt_file=str(netD_drs_ckpt_path),
        netD_drs=netD_drs,
        optD_drs=optD_drs,
        dataloader_drs=dl_drs,
        n_dis=args.n_dis,
        num_steps=args.num_steps,
        save_steps=1000,
        lr_decay=args.decay,
        dataloader=dl_train,
        log_dir=output_dir,
        print_steps=10,
        device=device,
        topk=args.topk,
        gold=args.gold,
        gold_step=args.p1_step,
        save_logits=False,
    )
    trainer.train()

if __name__ == '__main__':
    main()