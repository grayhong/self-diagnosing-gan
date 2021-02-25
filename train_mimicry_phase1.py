import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.trainer import LogTrainer
from diagan.utils.plot import (
    print_num_params
)
from diagan.utils.settings import set_seed


def get_dataloader(dataset, batch_size=128):
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=True)
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--root", "-r", default="./dataset/cifar10", type=str, help="dataset dir")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="cifar10", type=str, help="exp name")
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_pack', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--download_dataset', action='store_true')
    parser.add_argument('--topk', action='store_true')
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--logit_save_steps', default=100, type=int)
    parser.add_argument('--decay', default='linear', type=str)
    parser.add_argument('--n_dis', default=5, type=int)
    parser.add_argument('--imb_factor', default=0.1, type=float)
    parser.add_argument('--celeba_class_attr', default='glass', type=str)
    parser.add_argument('--ckpt_step', type=int)
    parser.add_argument('--no_save_logits', action='store_true')
    parser.add_argument('--save_logit_after', default=30000, type=int)
    parser.add_argument('--stop_save_logit_after', default=60000, type=int)
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

    netG, netD, optG, optD = get_gan_model(
        dataset_name=args.dataset,
        model=args.model,
        loss_type=args.loss_type,
        topk=args.topk,
    )

    print_num_params(netG, netD)

    ds_train = get_predefined_dataset(
        dataset_name=args.dataset,
        root=args.root,
    )
    dl_train = get_dataloader(ds_train, batch_size=args.batch_size)

    if args.dataset == 'celeba':
        args.num_steps = 75000
        args.logit_save_steps = 100
        args.save_logit_after= 55000
        args.stop_save_logit_after= 60000

    if args.dataset == 'cifar10':
        args.num_steps = 50000
        args.logit_save_steps = 100
        args.save_logit_after= 35000
        args.stop_save_logit_after= 40000

    print(args)

    if args.ckpt_step:
        netG_ckpt_file = save_path / f'checkpoints/netG/netG_{args.ckpt_step}_steps.pth'
        netD_ckpt_file = save_path / f'checkpoints/netD/netD_{args.ckpt_step}_steps.pth'
    else:
        netG_ckpt_file = None
        netD_ckpt_file = None

    # Start training
    trainer = LogTrainer(
        output_path=save_path,
        logit_save_steps=args.logit_save_steps,
        netG_ckpt_file=netG_ckpt_file,
        netD_ckpt_file=netD_ckpt_file,
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=args.n_dis,
        num_steps=args.num_steps,
        save_steps=1000,
        lr_decay=args.decay,
        dataloader=dl_train,
        log_dir=output_dir,
        print_steps=10,
        device=device,
        topk=args.topk,
        save_logits=not args.no_save_logits,
        save_logit_after=args.save_logit_after,
        stop_save_logit_after=args.stop_save_logit_after,
        )
    trainer.train()

if __name__ == '__main__':
    main()