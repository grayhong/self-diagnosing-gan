import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from diagan.datasets.predefined import get_predefined_dataset
from diagan.models.convnets import SimpleConvNet
from diagan.utils.settings import set_seed
from diagan.utils.trainer import accuracy, AverageMeter

from torch.utils import data
from torchvision.transforms import transforms
from tqdm import tqdm


def get_color_mnist_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

def get_mnist_fmnist_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    return transform



def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

    return top1.avg


def train(model, tr_loader, optimizer):
    model.train()
    top1 = AverageMeter()
    tr_iter = tqdm(iter(tr_loader))
    for x, y, _, _ in tr_iter:
        x, y = x.cuda(), y.cuda()
        N = x.size(0)
        pred, _ = model(x)
        loss = F.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, = accuracy(pred, y, topk=(1,))
        top1.update(prec1.item(), N)
    return top1.avg


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--num_data', type = int, default = 10000)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def main():
    opt = parse_option()
    set_seed(opt.seed)
    transform = get_mnist_fmnist_transform()

    ds_train = get_predefined_dataset(
        dataset_name='mnist_fmnist',
        root='./dataset/mnist_fmnist',
        weights=None,
        major_ratio=0.5,
        num_data=opt.num_data
    )

    dataloader = data.DataLoader(
        dataset=ds_train,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    model = SimpleConvNet(num_channels = 1, num_labels = 20).cuda()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [opt.epochs * 3 // 7, opt.epochs * 6 // 7], gamma=0.1)
    print(f'train_biased_model - opt: {optimizer}, sched: {scheduler}')

    ckpt_path = Path(f'./exp_results/mnist-fmnist-convnet-{opt.num_data}-seed{opt.seed}')
    ckpt_path.mkdir(exist_ok=True, parents=True)

    for n in range(1, opt.epochs + 1):
        train_acc = train(model, dataloader, optimizer)
        print(f'[{n} / {opt.epochs}] train_acc: {train_acc}')

        if n % 10 == 0:
            torch.save(model.state_dict(), ckpt_path / f'ckpt_{n}.pt')


if __name__ == '__main__':
    main()
