
import torch
import torch.nn.functional as F
from diagan.models.mnist import MNIST_DCGAN_Generator
from diagan.models.inception import InceptionV3

import numpy as np
from tqdm import tqdm 
from copy import deepcopy
import time


def get_activations(images, model, batch_size=50, dims=2048, device='cpu', verbose=True):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : List of image files
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    num_batches = len(images) // batch_size

    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for i in range(num_batches):
        start_time = time.time()

        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

        if verbose:
            print("\rINFO: Propagated batch %d/%d (%.4f sec/batch)" \
                % (i+1, num_batches, time.time()-start_time), end="", flush=True)

    return pred_arr


class InclusiveMNISTDCGANGenerator(MNIST_DCGAN_Generator):
    def __init__(self, loss_type='ns', **kwargs):
        print(f"Load DCGAN InclusiveGenerator loss_type: {loss_type}")
        MNIST_DCGAN_Generator.__init__(self, loss_type=loss_type, **kwargs)
        self.loss_type = loss_type

        # hold real image feature vectors
        if 'num_data' in kwargs:
            self.num_data = kwargs['num_data']
        elif 'dataset' in kwargs:
            if kwargs['dataset'] == 'cifar10':
                self.num_data = 50000
            elif kwargs['dataset'] == 'celeba':
                self.num_data = 162770
            else:
                raise NotImplementedError
        else:
            raise ValueError
        
        if 'dataloader' in kwargs:
            self.dataloader = kwargs['dataloader']
        else:
            raise ValueError

        self.setting = False

    def get_setting(self, train=True):
        net = 'vgg'
        version = '0.1'
        self.pdist = torch.nn.PairwiseDistance(p=2)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.device)
        if train:
            self.register_train_dataset_feats()
        self.setting = True

    
    def register_train_dataset_feats(self, path='train_feats_itp_95_seed2.pkl'):
        print('\n\n\nNew File\n\n\n')
        print('register_train_dataset_feats')
        data_iter = iter(self.dataloader)
        feats = {}
        for data, _, _, idx in tqdm(data_iter):
            data_feats = get_activations(data, model=self.inception, device=self.device)
            data_feats = torch.from_numpy(data_feats)
            for i in range(len(idx)):
                feats[int(idx[i].item())] = data_feats[i]
        train_feats = feats
            
        sorted_train_feats_keys = sorted(train_feats.keys())
        self.stack_train_feats = None
        for k in sorted_train_feats_keys:
            add_train_feats = torch.unsqueeze(train_feats[k], 0)
            if self.stack_train_feats is None:
                self.stack_train_feats = add_train_feats
            else:
                self.stack_train_feats = torch.cat((self.stack_train_feats, add_train_feats), 0)
        print('register_train_dataset_feats done')


    def get_latent_dataset_feats(self, path='latent_feats.pkl'):
        print('get_latent_dataset_feats')
        bs = 128
        num_latent = self.num_data * 10
        latent_candidate = torch.randn((num_latent, self.nz)).to(self.device)
        latent_splits = torch.split(latent_candidate, bs)
        feats = None
        idx = 0
        with torch.no_grad():
            for z in tqdm(latent_splits):
                gen_data = self.forward(z).detach()
                data_feats = get_activations(gen_data, model=self.inception, device=self.device)
                data_feats = torch.from_numpy(data_feats)
                if feats is None:
                    feats = data_feats
                else:
                    feats = torch.cat((feats, data_feats), 0)
        print('get_latent_dataset_feats done')

        return feats, latent_candidate

    def get_min_latent_idxs(self, latent_feats, batch_size=50):
        stack_train_feats = self.stack_train_feats.to(self.device)
        latent_splits = torch.split(latent_feats, batch_size)
        min_idxs = None
        min_dists = None
        # Use cdist
        cnt = 0
        for s in tqdm(latent_splits):
            s = s.to(self.device)
            d = torch.cdist(stack_train_feats, s)
            tmp_min_dists, tmp_min_idxs = torch.min(d, dim=1)
            tmp_min_idxs = tmp_min_idxs + cnt
            
            if min_idxs is None:
                min_idxs = tmp_min_idxs
                min_dists = tmp_min_dists
            else:
                min_idxs = torch.where(torch.le(tmp_min_dists, min_dists), tmp_min_idxs, min_idxs)
                min_dists = torch.where(torch.le(tmp_min_dists, min_dists), tmp_min_dists, min_dists)
            cnt += len(s)
        return min_idxs



    def compute_nearest_latent(self, batch_size=64):
        latent_feats, latent_candidate = self.get_latent_dataset_feats()
        self.nearest_latent = latent_candidate[self.get_min_latent_idxs(latent_feats, batch_size=batch_size)]

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
        if not self.setting:
            self.get_setting()

        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        S = int(self.num_data / batch_size * 20)
        sigma = 0.05
        lamb = 10
        beta = 0.4 * lamb

        if global_step % S == 0:
            self.compute_nearest_latent()

        if scaler is None:
            # Produce fake images
            fake_images = self.generate_images(num_images=batch_size,
                                               device=device)

            # Compute output logit of D thinking image real
            output = netD(fake_images)

            # Compute adversarial loss
            advG = self.compute_gan_loss(output=output)
            
            # Sample two sets of images
            copy_dataloader = deepcopy(self.dataloader)
            copy_dataiter = iter(copy_dataloader)
            comp_batch1 = next(copy_dataiter)
            copm_batch2 = next(copy_dataiter)
            comp_data1, _, _, comp_idx1 = comp_batch1
            comp_data2, _, _, comp_idx2 = copm_batch2
            comp_data1 = comp_data1.to(device)
            comp_data2 = comp_data2.to(device)
            comp_feat1 = self.stack_train_feats[comp_idx1].to(device)
            comp_feat2 = self.stack_train_feats[comp_idx2].to(device)

            # Compute reconstruction loss
            nearest_latent1 = self.nearest_latent[comp_idx1]
            nearest_latent2 = self.nearest_latent[comp_idx2]
            noise1 = torch.normal(mean=torch.zeros_like(nearest_latent1),
                                    std=sigma*torch.ones_like(nearest_latent1))
            noise2 = torch.normal(mean=torch.zeros_like(nearest_latent2),
                                    std=sigma*torch.ones_like(nearest_latent2))
            nz1 = nearest_latent1 + noise1
            nz2 = nearest_latent2 + noise2
            nz1 = nz1.to(device)
            nz2 = nz2.to(device)
            gen1 = self.forward(nz1)
            gen2 = self.forward(nz2)
            gen1_feats = get_activations(gen1, model=self.inception, device=device, verbose=False)
            gen2_feats = get_activations(gen2, model=self.inception, device=device, verbose=False)
            gen1_feats = torch.from_numpy(gen1_feats).to(device)
            gen2_feats = torch.from_numpy(gen2_feats).to(device)
            reconsG = 0.5 * torch.mean(self.pdist(gen1_feats, comp_feat1) + self.pdist(gen2_feats, comp_feat2))
            
            # Compute interpolated loss
            alpha = torch.rand(len(nz1), device=device)
            alpha_reshape = torch.unsqueeze(alpha, 1)
            itp_z = torch.mul(alpha_reshape, nz1) + torch.mul(1-alpha_reshape, nz2)
            gen_itp = self.forward(itp_z)

            gen_itp_feats = get_activations(gen_itp, model=self.inception, device=device, verbose=False)
            gen_itp_feats = torch.from_numpy(gen_itp_feats).to(device)
            itpG = torch.mean(alpha * self.pdist(gen_itp_feats, comp_feat1) + \
                            (1-alpha) * self.pdist(gen_itp_feats, comp_feat2))

            # Backprop and update gradients
            errG = advG + lamb * reconsG + beta * itpG
            errG.backward()
            optG.step()

        else:
            raise NotImplementedError

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data
