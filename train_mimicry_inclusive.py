import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from diagan.datasets.predefined import (
    get_predefined_dataset
)
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.trainer import LogTrainer
from diagan.utils.plot import (
    plot_color_mnist_generator, print_num_params
)
from diagan.utils.settings import set_seed


def get_dataloader(dataset, batch_size=128, clip=False, weights=None):
    if weights is not None:
        eps = 1e-1
        if clip:
            mean = weights.mean()
            var = weights.var()
            k = 2
            upper_bound = mean + k * var
            lower_bound = max(mean - k * var, eps)
            weight_list = np.array([lower_bound if i < lower_bound else (upper_bound if i > upper_bound else i) for i in weights])
        else:
            weight_list = np.array([eps if i < eps else i for i in weights])
        sampler = data.WeightedRandomSampler(weight_list, len(weight_list), replacement=True)
        print(f'weight_list max: {weight_list.max()} min: {weight_list.min()} mean: {weight_list.mean()} var: {weight_list.var()}')
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
    parser.add_argument("--dataset", "-d", default="color_mnist", type=str)
    parser.add_argument("--root", "-r", default="./dataset/colour_mnist", type=str, help="dataset dir")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="colour_mnist", type=str, help="exp name")
    parser.add_argument("--loss_type", default="ns", type=str, help="loss type")
    parser.add_argument("--model", default="mnist_dcgan", type=str, help="network model")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_pack', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--use_clipping', action='store_true')
    parser.add_argument('--num_steps', default=20000, type=int)
    parser.add_argument('--logit_save_steps', default=100, type=int)
    parser.add_argument('--decay', default='None', type=str)
    parser.add_argument('--n_dis', default=1, type=int)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--topk', default=0, type=int)
    parser.add_argument('--resample_score', type=str)
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



    ds_train = get_predefined_dataset(
        dataset_name=args.dataset,
        root=args.root,
        weights=None,
        major_ratio=args.major_ratio,
        num_data=args.num_data
    )
    dl_train = get_dataloader(
        ds_train,
        batch_size=args.batch_size,
        weights=None)

    netG, netD, optG, optD = get_gan_model(
        dataset_name=args.dataset,
        model=args.model,
        num_pack=args.num_pack,
        loss_type=args.loss_type,
        topk=args.topk == 1,
        inclusive=True,
        num_data=args.num_data,
        dataloader=dl_train,
    )

    print_num_params(netG, netD)

    print(args)

    # Start training
    trainer = LogTrainer(
        output_path=save_path,
        logit_save_steps=args.logit_save_steps,
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=args.n_dis,
        num_steps=args.num_steps,
        save_steps=1000,
        vis_steps=100,
        lr_decay=args.decay,
        dataloader=dl_train,
        log_dir=output_dir,
        print_steps=10,
        device=device,
        topk=args.topk,
        save_logits=args.num_pack==1,
        save_eval_logits=False,)
    trainer.train()

    plot_color_mnist_generator(netG, save_path=save_path, file_name='eval_p1')

    # if args.num_pack == 1:
    #     score_dict = calculate_scores(trainer.logit_results['netD_train'], start_epoch=args.num_steps // 2, end_epoch=args.num_steps)
    #     plot_score_sort(ds_train, score_dict, save_path=save_path, phase='p1')
    

if __name__ == '__main__':
    main()