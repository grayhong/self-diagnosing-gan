
import torch
import torch.nn.functional as F
from torch_mimicry.nets import sngan
from torch_mimicry.nets import infomax_gan
from torch_mimicry.nets import ssgan

def _bce_loss_with_logits(output, labels, **kwargs):
    r"""
    Wrapper for BCE loss with logits.
    """
    return F.binary_cross_entropy_with_logits(output, labels, reduction='none', **kwargs)


class TopKGenerator:
    def __init__(self, use_topk=False, decay_steps=2000):
        self.use_topk = use_topk
        self.topk_rate = 1
        self.decay_rate = 0.99
        self.decay_steps = 2000 # Unused
        self.min_topk_rate = 0.5

    def decay_topk_rate(self, step, epoch_steps=None):
        assert self.use_topk
        if epoch_steps:
            epoch = step // epoch_steps
        else:
            epoch = step // self.decay_steps
        self.topk_rate = max(self.decay_rate ** epoch, self.min_topk_rate)

    def get_topk(self, x, return_index=False):
        N = x.size(0)
        k = int(self.topk_rate * N)
        x, idx = torch.topk(x, k=k, dim=0)
        if return_index:
            return x, idx
        else:
            return x

class TopkSNGANGenerator32(sngan.SNGANGenerator32, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        sngan.SNGANGenerator32.__init__(self, **kwargs)
        print(f"Load SNGAN32 model topk: {topk} loss: {self.loss_type}")

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

class TopkSNGANGenerator64(sngan.SNGANGenerator64, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        sngan.SNGANGenerator64.__init__(self, **kwargs)
        print(f"Load SNGAN64 model topk: {topk} loss: {self.loss_type}")

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

class TopkInfoMaxGANGenerator32(infomax_gan.InfoMaxGANGenerator32, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        infomax_gan.InfoMaxGANGenerator32.__init__(self, **kwargs)
        print(f"Load InfoMaxGANGenerator32 model topk: {topk} loss: {self.loss_type}")

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        # Zero gradient every step.
        self.zero_grad()

        # Get only batch size from real batch
        real_images, _, _ = real_batch
        batch_size = real_images.shape[0]

        # Produce fake images
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        # Get logits and projected features
        output_fake, local_feat_fake, global_feat_fake = netD(fake_images)

        _, idx = self.get_topk(output_fake, return_index=True)

        idx = idx.view(-1)
        output_fake = output_fake[idx]
        local_feat_fake = local_feat_fake[idx]
        global_feat_fake = global_feat_fake[idx]

        local_feat_fake, global_feat_fake = netD.project_features(
            local_feat=local_feat_fake, global_feat=global_feat_fake)

        # Compute losses
        errG = self.compute_gan_loss(output_fake)

        errG_IM = netD.compute_infomax_loss(local_feat=local_feat_fake,
                                            global_feat=global_feat_fake,
                                            scale=self.infomax_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_IM

        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_IM', errG_IM, group='loss_IM')

        return log_data

class TopkInfoMaxGANGenerator64(infomax_gan.InfoMaxGANGenerator64, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        infomax_gan.InfoMaxGANGenerator64.__init__(self, **kwargs)
        print(f"Load InfoMaxGANGenerator64 model topk: {topk} loss: {self.loss_type}")

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        # Zero gradient every step.
        self.zero_grad()

        # Get only batch size from real batch
        real_images, _, _ = real_batch
        batch_size = real_images.shape[0]

        # Produce fake images
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        # Get logits and projected features
        output_fake, local_feat_fake, global_feat_fake = netD(fake_images)

        _, idx = self.get_topk(output_fake, return_index=True)

        idx = idx.view(-1)
        output_fake = output_fake[idx]
        local_feat_fake = local_feat_fake[idx]
        global_feat_fake = global_feat_fake[idx]

        local_feat_fake, global_feat_fake = netD.project_features(
            local_feat=local_feat_fake, global_feat=global_feat_fake)

        # Compute losses
        errG = self.compute_gan_loss(output_fake)

        errG_IM = netD.compute_infomax_loss(local_feat=local_feat_fake,
                                            global_feat=global_feat_fake,
                                            scale=self.infomax_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_IM

        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_IM', errG_IM, group='loss_IM')

        return log_data


class TopkSSGANGenerator32(ssgan.SSGANGenerator32, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        ssgan.SSGANGenerator32.__init__(self, **kwargs)
        print(f"Load SSGANGenerator32 model topk: {topk} loss: {self.loss_type}")

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and logits
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)
        output, _ = netD(fake_images)

        output = self.get_topk(output)
        # Compute GAN loss, upright images only.
        errG = self.compute_gan_loss(output)

        # Compute SS loss, rotates the images.
        errG_SS, _ = netD.compute_ss_loss(images=fake_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data


class TopkSSGANGenerator64(ssgan.SSGANGenerator64, TopKGenerator):
    def __init__(self, topk=False, **kwargs):
        TopKGenerator.__init__(self, use_topk=topk)
        ssgan.SSGANGenerator64.__init__(self, **kwargs)
        print(f"Load SSGANGenerator64 model topk: {topk} loss: {self.loss_type}")

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and logits
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)
        output, _ = netD(fake_images)

        output = self.get_topk(output)
        # Compute GAN loss, upright images only.
        errG = self.compute_gan_loss(output)

        # Compute SS loss, rotates the images.
        errG_SS, _ = netD.compute_ss_loss(images=fake_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data
