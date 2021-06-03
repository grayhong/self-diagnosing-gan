import argparse
import os
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models

from diagan.models.predefined_models import get_gan_model
from diagan.utils.settings import set_seed
from diagan.trainer.evaluate import DRS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")
    parser.add_argument("--exp_name", default="mimicry_pretrained-seed1", type=str, help="exp name")
    parser.add_argument("--model", default="sngan", type=str, help="network model")
    parser.add_argument("--loss_type", default="hinge", type=str, help="loss type")
    parser.add_argument("--classifier", default="vgg16", type=str, help="calssifier network model")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--netG_ckpt_step", type=int)
    parser.add_argument("--netG_train_mode", action='store_true')    
    parser.add_argument("--use_original_netD", action='store_true')
    parser.add_argument('--attr', default='Bald', type=str)
    parser.add_argument('--drs', action='store_true')
    parser.add_argument('--num_samples', default=50000, type=int)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    set_seed(args.seed)
    print(args)

    save_path = f'{args.work_dir}/{args.exp_name}'

    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    else:
        device = "cpu"

    # load model
    assert args.netG_ckpt_step
    print(f'load model from {save_path} step: {args.netG_ckpt_step}')
    if args.drs:
        netG, _, netD_drs, _, _, _ = get_gan_model(
            dataset_name='celeba',
            model=args.model,
            loss_type=args.loss_type,
            drs=True
        )
    else:
        netG, _, _, _ = get_gan_model(
            dataset_name='celeba',
            model=args.model,
            loss_type=args.loss_type,
        )
    netG.to(device)
    if not args.netG_train_mode:
        netG.eval()
        netG.to(device)
        if args.drs:
            netD_drs.eval()
            netD_drs.to(device)

    gan_ckpt = f'{args.work_dir}/{args.exp_name}/checkpoints/netG/netG_{args.netG_ckpt_step}_steps.pth'
    if args.use_original_netD:
        netD_drs_ckpt = f'{args.work_dir}/{args.exp_name}/checkpoints/netD/netD_{args.netG_ckpt_step}_steps.pth'
    else:
        netD_drs_ckpt = f'{args.work_dir}/{args.exp_name}/checkpoints/netD_drs/netD_drs_{args.netG_ckpt_step}_steps.pth'
    print(gan_ckpt)
    netG.restore_checkpoint(ckpt_file=gan_ckpt)
    if args.drs:
        netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt)
        netG = DRS(netG=netG, netD=netD_drs, device=device)

    # load classifier
    print('Load classifier')
    if args.classifier == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.classifier == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.classifier == 'inception':
        model = models.inception_v3(pretrained=True)
    else:
        raise ValueError('model should be vgg16 or resnet18 or inception')

    # change the number of classes 
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2, bias=True)

    classifier_path = './convnet_celeba'
    model.load_state_dict(torch.load(os.path.join(classifier_path, f'{args.attr}.pth')))
    model.to(device)

    batch_size = min(args.batch_size, args.num_samples)
    num_batches = args.num_samples // batch_size

    attr_num = 0
    not_attr_num = 0
    for i in range(num_batches):
        with torch.no_grad():
            img = netG.generate_images(batch_size, device=device)
            labels = model(img)
            answers = torch.argmax(labels, dim=1)
            attr = torch.count_nonzero(answers).item()
            not_attr = batch_size - attr
            attr_num += attr
            not_attr_num += not_attr

    print(f'attr: {attr_num}')
    print(f'not attr: {not_attr_num}')

    output_dir = os.path.join(save_path, 'evaluate', f'step-{args.netG_ckpt_step}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'count_attribute.csv')
    if os.path.exists(output_file):
        with open(output_file, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([args.attr, attr_num, not_attr_num])
    else:
        with open(output_file, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['', 'attr', 'not attr'])
            wr.writerow([args.attr, attr_num, not_attr_num])

if __name__ == '__main__':
    main()