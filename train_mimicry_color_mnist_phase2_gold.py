import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from diagan.datasets.predefined import (
    get_predefined_dataset
)
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.trainer import LogTrainer
from diagan.utils.plot import (
    plot_color_mnist_generator, plot_data,
                                    print_num_params
)
from diagan.utils.settings import set_seed


def get_dataloader(dataset, batch_size=128, clip=False, weights=None):
    if weights is not None:
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
        print(f'weight_list max: {weights.max()} min: {weights.min()} mean: {weights.mean()} var: {weights.var()}')
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
    parser.add_argument("--baseline_exp_name", default="colour_mnist", type=str, help="exp name")
    parser.add_argument("--model", default="mnistgan", type=str, help="network model")
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
    parser.add_argument('--p1_step', default=10000, type=int)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--resample_score', type=str)
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    baseline_output_dir = f'{args.work_dir}/{args.baseline_exp_name}'
    baseline_save_path = Path(baseline_output_dir)

    prefix = args.exp_name.split('/')[-1]

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
        gold=True
    )

    netG_ckpt_path = baseline_save_path / f'checkpoints/netG/netG_{args.p1_step}_steps.pth'
    netD_ckpt_path = baseline_save_path / f'checkpoints/netD/netD_{args.p1_step}_steps.pth'

    print_num_params(netG, netD)

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

    data_iter = iter(dl_train)
    imgs, _, _, _ = next(data_iter)
    plot_data(imgs, num_per_side=8, save_path=save_path, file_name=f'{prefix}_gold_train_data_p2', vis=None)

    print(args, netG_ckpt_path, netD_ckpt_path)

    # Start training
    trainer = LogTrainer(
        output_path=save_path,
        logit_save_steps=args.logit_save_steps,
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        netG_ckpt_file=netG_ckpt_path,
        netD_ckpt_file=netD_ckpt_path,
        n_dis=args.n_dis,
        num_steps=args.num_steps,
        save_steps=1000,
        vis_steps=100,
        lr_decay=args.decay,
        dataloader=dl_train,
        log_dir=output_dir,
        print_steps=10,
        device=device,
        save_logits=False,
        gold=True,
        gold_step=args.p1_step
    )
    trainer.train()

    plot_color_mnist_generator(netG, save_path=save_path, file_name=f'{prefix}-eval_p2')

    netG.restore_checkpoint(ckpt_file=netG_ckpt_path)
    netG.to(device)
    plot_color_mnist_generator(netG, save_path=save_path, file_name=f'{prefix}-eval_generated_p1')

    # score_dict = calculate_scores(trainer.logit_results['netD_train'], start_epoch=max(args.p1_step, args.num_steps // 2), end_epoch=args.num_steps)
    # plot_score_sort(ds_train, score_dict, save_path=save_path, phase='p2', plot_metric_name=args.resample_score)


if __name__ == '__main__':
    main()