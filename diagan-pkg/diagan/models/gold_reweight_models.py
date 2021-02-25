
import torch
import torch.nn.functional as F
from torch_mimicry.nets import sngan
from torch_mimicry.nets import infomax_gan
from torch_mimicry.nets import ssgan
from torch_mimicry.modules.losses import hinge_loss_dis, minimax_loss_dis
from torch_mimicry.nets.gan import gan

def compute_gold_reweight(output_fake ,d=1, eps=1e-6):
    with torch.no_grad():
        gold = output_fake**d
    return gold

def _bce_loss_with_logits(output, labels, **kwargs):
    r"""
    Wrapper for BCE loss with logits.
    """
    return F.binary_cross_entropy_with_logits(output, labels, reduction='none', **kwargs)

def gold_reweighted_minimax_loss_dis(output_fake,
                                     output_real, 
                                     real_label_val=1.0,
                                     fake_label_val=0.0,
                                     **kwargs):
    fake_weights = compute_gold_reweight(output_fake)

    # Produce real and fake labels.
    fake_labels = torch.full((output_fake.shape[0], 1),
                             fake_label_val,
                             device=output_fake.device)
    real_labels = torch.full((output_real.shape[0], 1),
                             real_label_val,
                             device=output_real.device)

    # FF, compute loss and backprop D
    errD_fake = _bce_loss_with_logits(output=output_fake,
                                      labels=fake_labels,
                                      **kwargs)
    weighted_errD_fake = fake_weights.view(-1) * errD_fake.view(-1)
    errD_fake = torch.mean(weighted_errD_fake)

    errD_real = _bce_loss_with_logits(output=output_real,
                                      labels=real_labels,
                                      **kwargs)
    errD_real = torch.mean(errD_real)

    # Compute cumulative error
    loss = errD_real + errD_fake

    return loss


def gold_reweighted_hinge_loss_dis(output_fake, output_real):
    fake_weights = compute_gold_reweight(output_fake)
    fake_out = F.relu(1.0 + output_fake)
    weighted_fake_out = fake_weights.view(-1) * fake_out.view(-1)
    loss = (F.relu(1.0 - output_real)).mean() + \
           (weighted_fake_out).mean()

    return loss

class GoldDiscriminator(gan.BaseDiscriminator):
    def __init__(self, loss_type='ns', **kwargs):
        print(f"Load GoldDiscriminator loss_type: {loss_type}")
        self.loss_type = loss_type

    def compute_gan_loss(self, output_real, output_fake):
        assert self.loss_type in ["hinge", "ns"]
        gold_loss_dict = {
            'hinge': gold_reweighted_hinge_loss_dis,
            'ns': gold_reweighted_minimax_loss_dis,
        }
        return gold_loss_dict[self.loss_type](output_fake=output_fake, output_real=output_real)

class GoldSNGANDiscriminator32(GoldDiscriminator, sngan.SNGANDiscriminator32):
    def __init__(self, **kwargs):
        print("Load SNGAN32 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        sngan.SNGANDiscriminator32.__init__(self, **kwargs)

class GoldSNGANDiscriminator64(GoldDiscriminator, sngan.SNGANDiscriminator64):
    def __init__(self, **kwargs):
        print("Load SNGAN64 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        sngan.SNGANDiscriminator64.__init__(self, **kwargs)

class GoldInfoMaxGANDiscriminator32(GoldDiscriminator, infomax_gan.InfoMaxGANDiscriminator32):
    def __init__(self, **kwargs):
        print("Load InfoMaxGAN32 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        infomax_gan.InfoMaxGANDiscriminator32.__init__(self, **kwargs)

class GoldInfoMaxGANDiscriminator64(GoldDiscriminator, infomax_gan.InfoMaxGANDiscriminator64):
    def __init__(self, **kwargs):
        print("Load InfoMaxGAN64 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        infomax_gan.InfoMaxGANDiscriminator64.__init__(self, **kwargs)

class GoldSSGANDiscriminator32(GoldDiscriminator, ssgan.SSGANDiscriminator32):
    def __init__(self, **kwargs):
        print("Load SSGAN32 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        ssgan.SSGANDiscriminator32.__init__(self, **kwargs)

class GoldSSGANDiscriminator64(GoldDiscriminator, ssgan.SSGANDiscriminator64):
    def __init__(self, **kwargs):
        print("Load SSGAN64 GOLD model")
        GoldDiscriminator.__init__(self, **kwargs)
        ssgan.SSGANDiscriminator64.__init__(self, **kwargs)