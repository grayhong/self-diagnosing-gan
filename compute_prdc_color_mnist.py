import argparse
import os
from pathlib import Path
import numpy as np
import torch


from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.convnets import SimpleConvNet
from diagan.utils.settings import set_seed

from diagan.datasets.predefined import get_predefined_dataset

from diagan.models.predefined_models import get_gan_model

from diagan.trainer.compute_pr import compute_pr
from diagan.trainer.evaluate import DRS
from diagan.utils.settings import set_seed
from torch.utils import data
from tqdm import tqdm, trange
import csv

def get_features(model, dataloader):
    model.eval()
    with torch.no_grad():
        data_iter = tqdm(iter(dataloader))
        num_data = len(dataloader.dataset)
        all_feats = torch.zeros(num_data, model.dim_in)
        for img, _, _, idx in data_iter:
            all_feats[idx] = model(img.cuda())[1].cpu()
    return all_feats

def get_fake_features(model, netG, num_samples=10000):
    model.eval()
    bs = 128
    with torch.no_grad():
        all_feats = torch.zeros(num_samples, model.dim_in)
        for i in trange(0, num_samples, bs):
            ns = min(num_samples - i, bs)
            fake_samples = netG.generate_images(ns)
            all_feats[torch.arange(i, i + ns)] = model(fake_samples.cuda())[1].cpu()
    return all_feats

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="colour_mnist", type=str, help="exp name")
    parser.add_argument("--model", default="mnist_dcgan", type=str, help="network model")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--drs', action='store_true')
    parser.add_argument('--file_name', default = 'prdc', type = str)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    return opt

def main():
    opt = parse_option()
    set_seed(opt.seed)
    print(opt)
    model = SimpleConvNet(num_labels = 20)
    ckpt_path = Path(f'./exp_results/color-mnist-convnet-60000-seed1')
    model.load_state_dict(torch.load(ckpt_path / 'ckpt_50.pt'))
    model = model.cuda()
    print(f'Load model from {ckpt_path}')
    ds_train = get_predefined_dataset(
        dataset_name='color_mnist',
        root='./dataset/colour_mnist',
        weights=None,
        major_ratio=opt.major_ratio,
        num_data=opt.num_data
    )
    dataloader = data.DataLoader(
        dataset=ds_train,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    real_feats = get_features(model, dataloader)
    num_fake = 10000 #round(opt.num_data*(1-opt.major_ratio))
    if opt.drs:
        netG, netD, netD_drs, optG, optD, optD_drs = get_gan_model(
            dataset_name='color_mnist',
            model=opt.model,
            num_pack=1,
            loss_type='ns',
            topk=False,
            drs=True
        )
    else:
        netG, netD, optG, optD = get_gan_model(
            dataset_name='color_mnist',
            model=opt.model,
            num_pack=1,
            loss_type='ns',
            topk=False,
        )
    gan_ckpt = f'{opt.work_dir}/{opt.exp_name}/checkpoints/netG/netG_20000_steps.pth'
    netD_drs_ckpt = f'{opt.work_dir}/{opt.exp_name}/checkpoints/netD_drs/netD_drs_20000_steps.pth'
    print(gan_ckpt)
    netG.restore_checkpoint(ckpt_file=gan_ckpt, optimizer=optG)
    if opt.drs:
        netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt, optimizer=optD_drs)
        netG = DRS(netG=netG, netD=netD_drs, device='cpu')
    fake_feats = get_fake_features(model, netG, num_fake)
    num_minor = sum(ds_train.dataset.biased_targets == 1)

    max_length = 10000
    if len(real_feats) > max_length:
        choice_index = np.random.choice(np.arange(len(real_feats)), size=max_length, replace=False)
        all_dict = compute_pr(real_feats[choice_index].cpu().numpy(), fake_feats.cpu().numpy(), nearest_k=3, device='cuda')
    else:
        all_dict = compute_pr(real_feats.cpu().numpy(), fake_feats.cpu().numpy(), nearest_k=3, device='cuda')
    major_dict = compute_pr(real_feats[ds_train.dataset.biased_targets == 0][:num_minor].cpu().numpy(), fake_feats.cpu().numpy(), nearest_k=3, device='cuda')
    minor_dict = compute_pr(real_feats[ds_train.dataset.biased_targets == 1].cpu().numpy(), fake_feats.cpu().numpy(), nearest_k=3, device='cuda')

    print(f"Major: {major_dict}")
    print(f"Minor: {minor_dict}")
    save_path = f'{opt.work_dir}/color_mnist_prdc'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_file = os.path.join(save_path, f'{opt.file_name}.csv')
    if os.path.exists(csv_file):
        f = open(csv_file, 'a', newline='')
        wr = csv.writer(f)
    else:
        f = open(csv_file, 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['Ratio', 'Seed','', 'Precision', 'Recall', 'Density', 'Coverage'])
    dict_dict = {'All': all_dict, 'Major': major_dict, 'Minor': minor_dict}
    for key, item_dict in dict_dict.items():
        prdc = list(item_dict.values())
        wr.writerow([opt.major_ratio, opt.seed, key]+ prdc)

if __name__ == '__main__':
    main()