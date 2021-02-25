
import torch
import torch.nn as nn
from torch_mimicry.nets.gan.gan import BaseDiscriminator, BaseGenerator

from diagan.models.gold_reweight_models import (
    gold_reweighted_hinge_loss_dis, gold_reweighted_minimax_loss_dis)
from diagan.models.topk_models import (
    TopKGenerator
)
from torch_mimicry.modules.losses import hinge_loss_dis, minimax_loss_dis


def get_norm(use_sn):
    if use_sn:  # spectral normalization
        return nn.utils.spectral_norm
    else:  # identity mapping
        return lambda x: x


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def weights_init_3channel(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def onehot(y, class_num):
    eye = torch.eye(class_num).type_as(y)  # ny x ny
    onehot = eye[y.view(-1)].float()  # B -> B x ny
    return onehot

# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class MNIST_DCGAN_Generator(BaseGenerator, TopKGenerator):
    def __init__(self, nz=100, nc=3, loss_type='hinge', topk=1, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk, decay_steps=10)
        BaseGenerator.__init__(self, nz=100,
                         ngf=128,
                         bottom_width=4,
                         loss_type=loss_type)
        print(f"Load MNIST_DCGAN_Generator reweight model loss_type: {loss_type} topk: {topk}")
        self.nz = nz
        self.fc = nn.Linear(nz, 384)
        self.tconv = nn.Sequential(
            # tconv1
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            # tconv2
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # tconv3
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            # tconv4
            nn.ConvTranspose2d(48, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        weights_init_3channel(self)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 384, 1, 1)
        x = self.tconv(x)
        return x

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   scaler=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        if scaler is None:
            # Produce fake images
            fake_images = self.generate_images(num_images=batch_size,
                                               device=device)

            # Compute output logit of D thinking image real
            output = netD(fake_images)

            output = self.get_topk(output)

            # Compute loss
            errG = self.compute_gan_loss(output=output)

            # Backprop and update gradients
            errG.backward()
            optG.step()

        else:
            with torch.cuda.amp.autocast():
                # Produce fake images
                fake_images = self.generate_images(num_images=batch_size,
                                                   device=device)

                # Compute output logit of D thinking image real
                output = netD(fake_images)

                output = self.get_topk(output)

                # Compute loss
                errG = self.compute_gan_loss(output=output)

            # Backprop and update gradients
            scaler.scale(errG).backward()
            scaler.step(optG)
            scaler.update()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data

# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class MNIST_DCGAN_Discriminator(BaseDiscriminator):
    def __init__(self, nc=3, num_pack=1, use_sn=False, loss_type='hinge', use_gold=False, **kwargs):
        print(f"Load MNIST_DCGAN_Discriminator reweight model loss_type {loss_type}, num_pack: {num_pack}, use_gold: {use_gold}")
        BaseDiscriminator.__init__(self, ndf=128, loss_type=loss_type)
        norm = get_norm(use_sn)
        self.num_pack = num_pack
        self.conv = nn.Sequential(
            # conv1
            norm(nn.Conv2d(nc * self.num_pack, 16, 3, 2, 1, bias=False)),  # use spectral norm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv2
            norm(nn.Conv2d(16, 32, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv3
            norm(nn.Conv2d(32, 64, 3, 2, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv4
            norm(nn.Conv2d(64, 128, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv5
            norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv6
            norm(nn.Conv2d(256, 512, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        self.out_d = nn.Linear(4 * 4 * 512, 1)
        weights_init_3channel(self)

        self.use_gold = use_gold

    def compute_gan_loss(self, output_real, output_fake):
        assert self.loss_type in ["hinge", "ns"]
        gold_loss_dict = {
            'hinge': gold_reweighted_hinge_loss_dis,
            'ns': gold_reweighted_minimax_loss_dis,
        }
        loss_dict = {
            'hinge': hinge_loss_dis,
            'ns': minimax_loss_dis,
        }
        if self.use_gold:
            return gold_loss_dict[self.loss_type](output_fake=output_fake, output_real=output_real)
        else:
            return loss_dict[self.loss_type](output_fake=output_fake, output_real=output_real)


    def forward(self, x, get_feature=False):
        batch_size = x.size(0)
        splitted_inputs = torch.split(x, int(batch_size / self.num_pack))
        packed_inputs = torch.cat(splitted_inputs, dim=1)
        
        x = self.conv(packed_inputs)
        x = x.view(-1, 4*4*512)
        if get_feature:
            return x
        else:
            return self.out_d(x)