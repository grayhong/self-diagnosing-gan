import argparse
import os
import pickle
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

from diagan.datasets.predefined import (
    get_predefined_dataset
)
from diagan.models import drs
from diagan.models.predefined_models import get_gan_model
from diagan.trainer.trainer import LogTrainer
from diagan.utils.plot import (
    calculate_scores,
                                    plot_color_mnist_generator, plot_data,
                                    plot_score_sort, print_num_params
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
    parser.add_argument('--num_steps', default=20000, type=int)
    parser.add_argument('--logit_save_steps', default=100, type=int)
    parser.add_argument('--decay', default='None', type=str)
    parser.add_argument('--n_dis', default=1, type=int)
    parser.add_argument('--p1_step', default=10000, type=int)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--resample_score', type=str)
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument('--use_eval_logits', type=int)
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

    netG, netD, netD_drs, optG, optD, optD_drs = get_gan_model(
        dataset_name=args.dataset,
        model=args.model,
        drs=True,
        loss_type=args.loss_type,
    )

    netG_ckpt_path = baseline_save_path / f'checkpoints/netG/netG_{args.p1_step}_steps.pth'
    netD_ckpt_path = baseline_save_path / f'checkpoints/netD/netD_{args.p1_step}_steps.pth'
    netD_drs_ckpt_path = baseline_save_path / f'checkpoints/netD/netD_{args.p1_step}_steps.pth'

    logit_path = baseline_save_path / ('logits_netD_eval.pkl' if args.use_eval_logits == 1 else 'logits_netD_train.pkl')
    print(f'Use logit from: {logit_path}')
    logits = pickle.load(open(logit_path, "rb"))
    score_start_step = args.p1_step - 5000
    score_end_step = args.p1_step
    score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)
    sample_weights = score_dict[args.resample_score]
    print(f'sample_weights mean: {sample_weights.mean()}, var: {sample_weights.var()}, max: {sample_weights.max()}, min: {sample_weights.min()}')


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
        weights=sample_weights if args.resample_score is not None else None)
    dl_drs = get_dataloader(ds_train, batch_size=args.batch_size, weights=None)


    data_iter = iter(dl_train)
    imgs, _, _, _ = next(data_iter)
    plot_data(imgs, num_per_side=8, save_path=save_path, file_name=f'{prefix}_resampled_train_data_p2', vis=None)
    plot_score_sort(ds_train, score_dict, save_path=save_path, phase=f'{prefix}_{score_start_step}-{score_end_step}_score', plot_metric_name=args.resample_score)
    # plot_score_box(ds_train, score_dict, save_path=save_path, phase=f'{prefix}_{score_start_step}-{score_end_step}_box')

    print(args, netG_ckpt_path, netD_ckpt_path, netD_drs_ckpt_path)

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
        netD_drs_ckpt_file=netD_drs_ckpt_path,
        netD_drs=netD_drs,
        optD_drs=optD_drs,
        dataloader_drs=dl_drs,
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
    )
    trainer.train()

    plot_color_mnist_generator(netG, save_path=save_path, file_name=f'{prefix}-eval_p2')

    netG_drs = drs.DRS(netG, netD_drs, device=device)
    # for percentile in np.arange(50, 100, 5):
    # netG_drs.percentile = percentile
    percentile = 80
    plot_color_mnist_generator(netG_drs, save_path=save_path, file_name=f'{prefix}-eval_drs_percent{percentile}_p2')

    netG.restore_checkpoint(ckpt_file=netG_ckpt_path)
    netG.to(device)
    plot_color_mnist_generator(netG, save_path=save_path, file_name=f'{prefix}-eval_generated_p1')

    # score_dict = calculate_scores(trainer.logit_results['netD_train'], start_epoch=max(args.p1_step, args.num_steps // 2), end_epoch=args.num_steps)
    # plot_score_sort(ds_train, score_dict, save_path=save_path, phase='p2', plot_metric_name=args.resample_score)


if __name__ == '__main__':
    main()