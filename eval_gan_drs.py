import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from diagan.models.predefined_models import get_gan_model
from diagan.trainer.evaluate import evaluate_drs, evaluate_drs_ffhq
from diagan.utils.settings import set_seed



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--root", "-r", default="./dataset/cifar10", type=str, help="dataset dir")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--gpu', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--netG_ckpt_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')
    parser.add_argument("--use_original_netD", action='store_true')
    args = parser.parse_args()

    if args.gpu:
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
        loss_type=args.loss_type,
        drs=True,
    )
    if not args.netG_train_mode:
        netG.eval()
        netD_drs.eval()
        netG.to(device)
        netD_drs.to(device)

    if args.dataset == 'celeba':
        dataset = 'celeba_64'
    else:
        dataset = args.dataset

    if args.dataset == 'celeba':
        stats_name = 'celeba_64_202k_run_0'
    elif args.dataset == 'cifar10':
        stats_name = 'cifar10_train'
    elif args.dataset == 'ffhq':
        stats_name = 'ffhq_69k_run_0'
    stats_file = f'./precalculated_statistics/fid_stats_{stats_name}.npz'
    
    is_stylegan2 = args.model == 'stylegan2'


    print(args)
    if args.dataset == 'ffhq':
        # Evaluate fid
        evaluate_drs_ffhq(
            metric='fid',
            log_dir=save_path,
            netG=netG,
            netD_drs=netD_drs,
            dataset=dataset,
            num_real_samples=50000,
            num_fake_samples=50000,
            evaluate_step=args.netG_ckpt_step,
            num_runs=1,
            device=device,
            stats_file=stats_file,
            use_original_netD=args.use_original_netD,
            is_stylegan2=is_stylegan2
        )
    else:
        # Evaluate fid
        evaluate_drs(
            metric='fid',
            log_dir=save_path,
            netG=netG,
            netD_drs=netD_drs,
            dataset=dataset,
            num_real_samples=50000,
            num_fake_samples=50000,
            evaluate_step=args.netG_ckpt_step,
            num_runs=1,
            device=device,
            stats_file=stats_file,
            use_original_netD=args.use_original_netD,
            is_stylegan2=is_stylegan2
        )

        # Evaluate inception score
        evaluate_drs(
            metric='inception_score',
            log_dir=save_path,
            netG=netG,
            netD_drs=netD_drs,
            num_samples=50000,
            evaluate_step=args.netG_ckpt_step,
            num_runs=1,
            device=device,
            use_original_netD=args.use_original_netD,
            is_stylegan2=is_stylegan2
        )
        
        # Evaluate PR
        evaluate_drs(
            metric='pr',
            log_dir=save_path,
            netG=netG,
            netD_drs=netD_drs,
            dataset=dataset,
            num_real_samples=10000,
            num_fake_samples=10000,
            evaluate_step=args.netG_ckpt_step,
            num_runs=1,
            device=device,
            use_original_netD=args.use_original_netD,
            is_stylegan2=is_stylegan2
        )

if __name__ == '__main__':
    main()