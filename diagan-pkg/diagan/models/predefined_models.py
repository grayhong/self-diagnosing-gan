import torch.optim as optim
from diagan.models.gold_reweight_models import (
    GoldInfoMaxGANDiscriminator32,
    GoldInfoMaxGANDiscriminator64,
    GoldSNGANDiscriminator32, GoldSNGANDiscriminator64,
    GoldSSGANDiscriminator32, GoldSSGANDiscriminator64)
from diagan.models.inclusive_gan import InclusiveMNISTDCGANGenerator
from diagan.models.mnist import (MNIST_DCGAN_Discriminator, MNIST_DCGAN_Generator)
from diagan.models.stylegan2 import StyleGANDiscriminator, StyleGANGenerator
from diagan.models.topk_models import (TopkSNGANGenerator32, TopkSNGANGenerator64,
                                       TopkSSGANGenerator32, TopkSSGANGenerator64,
                                       TopkInfoMaxGANGenerator32, TopkInfoMaxGANGenerator64)
from diagan.models.toy import Toy_Discriminator, Toy_Generator
from torch_mimicry.nets import infomax_gan, sngan, ssgan


def get_cifar10_gen(model='sngan', loss_type='hinge', gold=False, topk=False, **kwargs):
    model_dict = {
        'sngan': sngan.SNGANGenerator32,
        'infomax_gan': infomax_gan.InfoMaxGANGenerator32,
        'ssgan': ssgan.SSGANGenerator32,
    }
    topk_model_dict = {
        'sngan': TopkSNGANGenerator32,
        'infomax_gan': TopkInfoMaxGANGenerator32,
        'ssgan': TopkSSGANGenerator32,
    }
    if topk:
        netG = topk_model_dict[model](loss_type=loss_type, topk=topk, **kwargs)
    else:
        netG = model_dict[model](loss_type=loss_type, **kwargs)
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))
    return netG, optG


def get_cifar10_disc(model='sngan', loss_type='hinge', gold=False, topk=False, **kwargs):
    model_dict = {
        'sngan': sngan.SNGANDiscriminator32,
        'infomax_gan': infomax_gan.InfoMaxGANDiscriminator32,
        'ssgan': ssgan.SSGANDiscriminator32,
    }
    gold_model_dict = {
        'sngan': GoldSNGANDiscriminator32,
        'infomax_gan': GoldInfoMaxGANDiscriminator32,
        'ssgan': GoldSSGANDiscriminator32
    }
    if gold:
        netD = gold_model_dict[model](loss_type=loss_type, **kwargs)
    else:
        netD = model_dict[model](loss_type=loss_type, **kwargs)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    return netD, optD


def get_celeba_gen(model='sngan', loss_type='hinge', gold=False, topk=False, **kwargs):
    model_dict = {
        'sngan': sngan.SNGANGenerator64,
        'infomax_gan': infomax_gan.InfoMaxGANGenerator64,
        'ssgan': ssgan.SSGANGenerator64,
    }
    topk_model_dict = {
        'sngan': TopkSNGANGenerator64,
        'infomax_gan': TopkInfoMaxGANGenerator64,
        'ssgan': TopkSSGANGenerator64,
    }
    if topk:
        netG = topk_model_dict[model](loss_type=loss_type, topk=topk, **kwargs)
    else:
        netG = model_dict[model](loss_type=loss_type, **kwargs)
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))
    return netG, optG


def get_celeba_disc(model='sngan', loss_type='hinge', gold=False, topk=False, **kwargs):
    model_dict = {
        'sngan': sngan.SNGANDiscriminator64,
        'infomax_gan': infomax_gan.InfoMaxGANDiscriminator64,
        'ssgan': ssgan.SSGANDiscriminator64,
    }
    gold_model_dict = {
        'sngan': GoldSNGANDiscriminator64,
        'infomax_gan': GoldInfoMaxGANDiscriminator64,
        'ssgan': GoldSSGANDiscriminator64
    }
    if gold:
        netD = gold_model_dict[model](loss_type=loss_type, **kwargs)
    else:
        netD = model_dict[model](loss_type=loss_type, **kwargs)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    return netD, optD


def get_toy_gen(model='toy', **kwargs):
    netG = Toy_Generator()
    optG = optim.Adam(netG.parameters(), 1e-4, betas=(0.5, 0.999))
    return netG, optG


def get_toy_disc(model='toy', **kwargs):
    netD = Toy_Discriminator()
    optD = optim.Adam(netD.parameters(), 1e-4, betas=(0.5, 0.999))
    return netD, optD


def get_color_mnist_gen(model='mnist_dcgan', reweight=False, loss_type='ns', gold=False, num_pack=1, topk=False,
                        **kwargs):
    model_dict = {
        'mnist_dcgan': MNIST_DCGAN_Generator,
    }
    if 'inclusive' in kwargs and kwargs['inclusive']:
        netG = InclusiveMNISTDCGANGenerator(loss_type=loss_type, topk=topk, **kwargs)
    else:
        netG = model_dict[model](loss_type=loss_type, topk=topk, **kwargs)
    optG = optim.Adam(netG.parameters(), 1e-4, betas=(0.5, 0.9))
    return netG, optG


def get_color_mnist_disc(model='mnist_dcgan', loss_type='hinge', gold=False, num_pack=1, topk=False, **kwargs):
    model_dict = {
        'mnist_dcgan': MNIST_DCGAN_Discriminator,
    }
    netD = model_dict[model](use_gold=gold, loss_type=loss_type, num_pack=num_pack, **kwargs)
    optD = optim.Adam(netD.parameters(), 1e-4, betas=(0.5, 0.9))
    return netD, optD


def get_mnist_fmnist_gen(model='mnist_dcgan', loss_type='hinge', gold=False, num_pack=1, topk=False, **kwargs):
    model_dict = {
        'mnist_dcgan': MNIST_DCGAN_Generator,
    }
    if 'inclusive' in kwargs and kwargs['inclusive']:
        netG = InclusiveMNISTDCGANGenerator(nc=1, loss_type=loss_type, topk=topk, **kwargs)
    else:
        netG = model_dict[model](nc=1, loss_type=loss_type, topk=topk, **kwargs)
    optG = optim.Adam(netG.parameters(), 1e-4, betas=(0.5, 0.9))
    return netG, optG


def get_mnist_fmnist_disc(model='mnist_dcgan', loss_type='hinge', gold=False, num_pack=1, topk=False, **kwargs):
    model_dict = {
        'mnist_dcgan': MNIST_DCGAN_Discriminator,
    }
    if model == 'sngan':
        if gold or topk or num_pack != 1:
            raise NotImplementedError
        netD = model_dict[model]()
    else:
        netD = model_dict[model](nc=1, use_gold=gold, loss_type=loss_type, num_pack=num_pack, **kwargs)
    optD = optim.Adam(netD.parameters(), 1e-4, betas=(0.5, 0.9))
    return netD, optD


def get_ffhq_gen(model='stylegan', **kwargs):
    netG = StyleGANGenerator(size=256, **kwargs)
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))
    return netG, optG


def get_ffhq_disc(model='stylegan', **kwargs):
    netD = StyleGANDiscriminator(size=256, **kwargs)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    return netD, optD


DATASET_DICT = {
    'celeba': (get_celeba_gen, get_celeba_disc),
    'cifar10': (get_cifar10_gen, get_cifar10_disc),
    '25gaussian': (get_toy_gen, get_toy_disc),
    'ffhq': (get_ffhq_gen, get_ffhq_disc),
    'color_mnist': (get_color_mnist_gen, get_color_mnist_disc),
    'mnist_fmnist': (get_mnist_fmnist_gen, get_mnist_fmnist_disc),
}


def get_gan_model(dataset_name, model='sngan', loss_type="hinge", gold=False, drs=False, **kwargs):
    netG_fn, netD_fn = DATASET_DICT[dataset_name]
    netG, optG = netG_fn(model=model, loss_type=loss_type, gold=gold, **kwargs)
    netD, optD = netD_fn(model=model, loss_type=loss_type, gold=gold, **kwargs)
    if drs:
        netD_drs, optD_drs = netD_fn(model=model, loss_type='ns', **kwargs)
        return netG, netD, netD_drs, optG, optD, optD_drs
    else:
        return netG, netD, optG, optD
