import argparse
import os
from pathlib import Path
from diagan.trainer.evaluate import evaluate_with_attr

import torch
import torch.backends.cudnn as cudnn

from diagan.models.predefined_models import get_gan_model
from diagan.utils.settings import set_seed

import csv

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="celeba", type=str)
    parser.add_argument("--root", "-r", default="./dataset/celeba", type=str, help="dataset dir")
    parser.add_argument("--attr", default="Bald", type=str, help="attribute name")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--netG_ckpt_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    return opt

def main():
    args = parse_option()
    set_seed(args.seed)
    print(args)

    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

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
    )
    netG.to(device)
    if not args.netG_train_mode:
        netG.eval()

    if args.dataset == 'celeba':
        dataset = 'celeba_64'
    else:
        raise ValueError("Dataset should be CelebA")

    evaluate_with_attr(metric='partial_recall',
                        attr=args.attr,
                        log_dir=save_path,
                        netG=netG,
                        dataset=dataset,
                        num_real_samples=10000,
                        num_fake_samples=10000,
                        evaluate_step=args.netG_ckpt_step,
                        num_runs=1,
                        device=device,)


if __name__ == '__main__':
    main()