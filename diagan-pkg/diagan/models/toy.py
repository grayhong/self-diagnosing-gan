
import torch.nn as nn

from torch_mimicry.nets.gan import gan


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


class Toy_Generator(gan.BaseGenerator):
	def __init__(self, nz=2, nc=2, dim=256):
		super().__init__(nz=nz,
                         ngf=256,
                         bottom_width=4,
                         loss_type='ns')
		self.net = nn.Sequential(
			nn.Linear(nz, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, nc),
		)
		weights_init(self)

	def forward(self, x):
		return self.net(x)

# https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
class Toy_Discriminator(gan.BaseDiscriminator):
	def __init__(self, nc=2, dim=256, use_sn=False):
		super().__init__(ndf=256,
                         loss_type='ns')
		norm = get_norm(use_sn)
		self.net = nn.Sequential(
			nn.Linear(nc, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim),
			nn.ReLU(True),
		)
		self.out_d = nn.Linear(dim, 1)
		weights_init(self)

	def forward(self, x):
		x = self.net(x)
		return self.out_d(x)