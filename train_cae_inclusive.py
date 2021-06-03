import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import dataloader
from tqdm import trange

from diagan.datasets.generated import (
    get_generated_dataset
)
from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.auto_encoder import get_ae_model
from diagan.models.drs import DRS
from diagan.models.predefined_models import get_gan_model
from diagan.utils.plot import (
    show_sorted_score_samples, to_numpy_image
)
from diagan.utils.settings import set_seed
from diagan.utils.trainer import AverageMeter, save_np_arr


def test_cae(data_loader, model, device='cuda'):
    model.eval()
    losses = []
    data_iter = iter(data_loader)
    loss_arr = np.zeros(len(data_loader.dataset))
    with torch.no_grad():
        for inputs, _, _, idx in data_iter:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_size = inputs.size(0)
            loss = outputs.sub(inputs).pow(2).view(batch_size, -1)
            loss = loss.sum(dim=1, keepdim=False).sqrt().div(32)
            loss_arr[idx.cpu().numpy()] = loss.detach().cpu().numpy()
    model.train()
    return loss_arr


def train_cae(model, dl_train, dl_test, save_path, epochs=100, device='cuda'):
    criterion = nn.MSELoss()
    print_steps = 10
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    losses = AverageMeter()
    
    logit_save_epochs = 1
    model_save_epochs = 50

    loss_epochs = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            data_iter = iter(dl_train)
            for batch_idx, (inputs, _) in enumerate(data_iter):
                inputs = inputs.to(device)
                outputs = model(inputs)

                loss = criterion(inputs, outputs)

                losses.update(loss.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch: [{} | {}], loss: {}'.format(epoch, epochs, losses.avg))

                    
            if epoch > 0 and epoch % model_save_epochs == 0:
                print(f'Saving model.. {epoch} / {epochs}')
                torch.save(model.state_dict(), save_path / f'cae_epoch-{epoch}.pth')
            if epoch > 0 and epoch % logit_save_epochs == 0:
                print(f'Saving loss.. {epoch} / {epochs}')
                loss_arr = test_cae(dl_test, model, device=device)
                loss_epochs.append(loss_arr)
    finally:
        loss_epoch_arr = np.stack(loss_epochs, axis=1)
        save_np_arr(loss_epoch_arr, save_path / f'cae_training_loss.npy')

    return model

def get_dataloader(dataset, batch_size=128):
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=True)
    return dataloader

def generate_dataset(netG, save_path, eval_mode=True, device='cuda', num_images=50000):
    print(f'Generate data... eval_mode: {eval_mode}')
    if eval_mode:
        netG.eval()
    with torch.no_grad():
        img_arr = []
        step_data = 1000
        num_epochs = num_images // step_data
        for t in trange(num_epochs):
            imgs = netG.generate_images(num_images=step_data, device=device).detach().cpu()
            imgs = to_numpy_image(imgs)
            img_arr.append(imgs)
        imgs = np.concatenate(img_arr, 0)
        pickle.dump(imgs, open(save_path, 'wb'))
        del imgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="cifar10", type=str)
    parser.add_argument("--root", "-r", default="./dataset/cifar10", type=str, help="dataset dir")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--netG_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')
    parser.add_argument("--cae_ckpt_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--loss_type", default='ns', type=str)
    parser.add_argument("--generated_dataset_path", type=str)
    parser.add_argument('--major_ratio', default=0.99, type=float)
    parser.add_argument('--num_data', default=10000, type=int)
    parser.add_argument('--num_pack', default=1, type=int)
    parser.add_argument('--topk', action='store_true')
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
    
    
    if args.dataset == 'mnist_c':
        ds_test = get_predefined_dataset(
            dataset_name=args.dataset,
            root=args.root,
        )
    else:
        ds_test = get_predefined_dataset(
            dataset_name=args.dataset,
            root=args.root,
            major_ratio=args.major_ratio,
            num_data=args.num_data
        )
    dl_test = get_dataloader(dataset=ds_test, batch_size=args.batch_size)

    # load model
    assert args.netG_step
    print(f'load model from: {args.netG_step}')
    netG, _, netD_drs, _, _, _ = get_gan_model(
        args.dataset, 
        model=args.model,
        drs=True, 
        loss_type=args.loss_type, 
        topk=args.topk, 
        num_pack=args.num_pack,
        inclusive=True,
        num_data=args.num_data,
        dataloader=dl_test,)
    netG.to(device)
    netD_drs.to(device)
    netG.get_setting(train=False)

    step = netG.restore_checkpoint(ckpt_file=save_path / f'checkpoints/netG/netG_{args.netG_step}_steps.pth')

    netD_drs_ckpt_path = save_path / f'checkpoints/netD_drs/netD_drs_{args.netG_step}_steps.pth'
    if os.path.exists(netD_drs_ckpt_path):
        use_drs = True
        netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt_path)
        netD_drs.to(device)
        drs = DRS(netG=netG, netD=netD_drs, device=device)
    else:
        use_drs = False
        drs = netG
        
    print(f'use drs: {use_drs}')
    


    model = get_ae_model(dataset_name=args.dataset).to(device)
    if args.cae_ckpt_path:
        model.load_state_dict(torch.load(args.cae_ckpt_path))
    else:
        if args.generated_dataset_path:
            print(f'skip data generation, use: {args.generated_dataset_path}')
            generated_dataset_path = args.generated_dataset_path
        else:
            # generate dataset
            generated_dataset_path = save_path / f'netG_{step}_steps_seed{args.seed}_generated_dataset.pkl'
            generate_dataset(drs, generated_dataset_path, eval_mode=not args.netG_train_mode, device=device)
            print(f'data generated in: {generated_dataset_path}')
        
        ds_train = get_generated_dataset(dataset_name=args.dataset, root=generated_dataset_path)
        dl_train = get_dataloader(dataset=ds_train, batch_size=args.batch_size)
        cae_ckpt_path = save_path / 'cae_checkpoints' / f'{step}_steps_seed{args.seed}'
        cae_ckpt_path.mkdir(parents=True, exist_ok=True)
        model = train_cae(model, dl_train=dl_train, dl_test=dl_test, save_path=cae_ckpt_path, epochs=args.epochs)

    final_loss = test_cae(dl_test, model)
    final_score = final_loss
    pickle.dump(final_score, open(save_path / f'netG_{step}_steps_seed{args.seed}_epoch{args.epochs}_ae_score.pkl', 'wb'))
    show_sorted_score_samples(
        dataset=ds_test,
        score=final_score,
        save_path=save_path,
        score_name='ae_score',
        plot_name=f'netG_{step}_steps_seed{args.seed}_epoch{args.epochs}_ae_score')
    
if __name__ == '__main__':
    main()