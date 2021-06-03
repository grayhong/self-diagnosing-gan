import argparse
import math
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from dataset import MultiResolutionDataset
from distributed import (get_rank, get_world_size, reduce_loss_dict,
                         reduce_sum, synchronize)
from diagan.utils.settings import set_seed
from diagan.utils.plot import calculate_scores
from model import Discriminator, Generator
from non_leaking import AdaptiveAugment, augment

try:
    import wandb

except ImportError:
    wandb = None



def data_sampler(dataset, shuffle, distributed, weights=None):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if weights is not None:
        return data.WeightedRandomSampler(weights, len(weights), replacement=True)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_logit(dataloader, netD, device):
    data_iter = iter(dataloader)
    logit_list = np.zeros(len(dataloader.dataset))
    # netD.eval()
    with torch.no_grad():
        for data, _, _, idx in data_iter:
            real_data = data.to(device)
            logit_r = netD(real_data).view(-1)
            logit_list[idx.cpu().numpy()] = logit_r.detach().cpu().numpy()
    netD.train()
    return logit_list

def save_logit(logits_dict, output_path):
    for name, logits in logits_dict.items():
        pickle.dump(logits, open(output_path / f'logits_{name}.pkl', 'wb'))
    
def train(args, loader, drs_loader, generator, discriminator, drs_discriminator, g_optim, d_optim, drs_d_optim, g_ema, device, output_path):
    # loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        drs_d_module = drs_discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        drs_d_module = drs_discriminator
    
    logit_results = defaultdict(dict)

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    iter_dataloader = iter(loader)
    iter_drs_dataloader = iter(drs_loader)
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        try:
            real_img, _ = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(loader)
            real_img, _ = next(iter_dataloader)

        try:
            drs_real_img, _ = next(iter_drs_dataloader)
        except StopIteration:
            iter_drs_dataloader = iter(drs_loader)
            drs_real_img, _ = next(iter_drs_dataloader)
        
        real_img = real_img.to(device)
        drs_real_img = drs_real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(drs_discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            drs_real_img_aug, _ = augment(drs_real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img
            drs_real_img_aug = drs_real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)

        drs_fake_pred = drs_discriminator(fake_img)
        drs_real_pred = drs_discriminator(drs_real_img_aug)

        d_loss = d_logistic_loss(real_pred, fake_pred)
        drs_d_loss = d_logistic_loss(drs_real_pred, drs_fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["drs_d"] = drs_d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        drs_discriminator.zero_grad()
        drs_d_loss.backward()
        drs_d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

            drs_real_img.requires_grad = True
            drs_real_pred = drs_discriminator(drs_real_img)
            drs_r1_loss = d_r1_loss(drs_real_pred, drs_real_img)

            drs_discriminator.zero_grad()
            (args.r1 / 2 * drs_r1_loss * args.d_reg_every + 0 * drs_real_pred[0]).backward()

            drs_d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(drs_discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        drs_d_loss_val = loss_reduced["drs_d"].mean().item()

        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0 and i > 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; drs_d: {drs_d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    save_path = output_path / 'fixed_sample'
                    save_path.mkdir(parents=True, exist_ok=True)
                    utils.save_image(
                        sample,
                        save_path / f"{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    random_sample_z = torch.randn(args.n_sample, args.latent, device=device)
                    sample, _ = g_ema([random_sample_z])
                    save_path = output_path / 'random_sample'
                    save_path.mkdir(parents=True, exist_ok=True)
                    utils.save_image(
                        sample,
                        save_path / f"{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )


            if i % 5000 == 0:
                save_path = output_path / 'checkpoint'
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "drs_d": drs_d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "drs_d_optim": drs_d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    save_path / f"{str(i).zfill(6)}.pt",
                )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--root", "-r", default="./dataset/cifar10", type=str, help="dataset dir")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=32, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="test", type=str, help="exp name")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--baseline_exp_name', type=str)
    parser.add_argument('--resample_score', type=str)
    parser.add_argument('--p1_step', default=200000, type=int)
    parser.add_argument('--logit_save_steps', default=100, type=int)
    parser.add_argument('--save_logit_after', default=1000000, type=int)
    # parser.add_argument('--stop_save_logit_after', default=45000, type=int)

    args = parser.parse_args()
    print(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    output_dir = f'{args.work_dir}/{args.exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.dataset == 'cifar10':
        args.size = 32
    elif args.dataset == 'celeba':
        args.size = 64
    elif args.dataset == 'utk_faces':
        args.size = 64
    elif args.dataset == 'imagenet':
        args.size = 128
    elif args.dataset == 'ffhq':
        args.size = 256
    else:
        raise AttributeError(f'{args.dataset} not supported')

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    drs_discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    drs_d_optim = optim.Adam(
        drs_discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = f'./exp_results/{args.baseline_exp_name}/checkpoint/{str(args.p1_step).zfill(6)}.pt'
    print("load model:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(ckpt_path)
        args.start_iter = int(os.path.splitext(ckpt_name)[0]) + 1

    except ValueError:
        pass

    generator.load_state_dict(ckpt["g"])
    discriminator.load_state_dict(ckpt["d"])
    drs_discriminator.load_state_dict(ckpt["d"])

    g_ema.load_state_dict(ckpt["g_ema"])

    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])
    drs_d_optim.load_state_dict(ckpt["d_optim"])
    print(f'start_iter: {args.start_iter}')

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        drs_discriminator = nn.parallel.DistributedDataParallel(
            drs_discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = MultiResolutionDataset(args.root, transform, args.size)


    logit_path = f'./exp_results/{args.baseline_exp_name}/logits_netD.pkl'
    print(f'Use logit from: {logit_path}')
    logits = pickle.load(open(logit_path, "rb"))
    
    window = 5000
    score_start_step = (args.p1_step - window)
    score_end_step = args.p1_step + 1
    score_dict = calculate_scores(logits, start_epoch=score_start_step, end_epoch=score_end_step)

    sample_weights = score_dict[args.resample_score]
    def print_stats(sw): 
        print(f'weight_list max: {sw.max()} min: {sw.min()} mean: {sw.mean()} var: {sw.var()}')
    print_stats(sample_weights)
    # for k, v in score_dict.items():
    #     print(k)
    #     print_stats(v)
    # import ipdb; ipdb.set_trace()

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed, weights=sample_weights),
        drop_last=True,
        num_workers=4,
    )

    drs_loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed, weights=None),
        drop_last=True,
        num_workers=4,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(
        args=args,
        loader=loader,
        drs_loader=drs_loader,
        generator=generator,
        discriminator=discriminator,
        drs_discriminator=drs_discriminator,
        g_optim=g_optim,
        d_optim=d_optim,
        drs_d_optim=drs_d_optim,
        g_ema=g_ema, 
        device=device,
        output_path=save_path)
