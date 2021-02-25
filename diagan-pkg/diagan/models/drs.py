

import numpy as np
import torch
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DRS(nn.Module):
    def __init__(self, netG, netD, device, gamma=None, percentile=80):
        super().__init__()
        self.netG = netG
        self.netD = netD
        self.maximum = -100000
        self.device = device
        self.batch_size = 256
        self.percentile = percentile
        self.gamma = gamma
        self.init_drs()
    
    def get_fake_samples_and_ldr(self, num_data):
        with torch.no_grad():
            imgs = self.netG.generate_images(num_data, device=self.device)
            netD_out = self.netD(imgs)
            if type(netD_out) is tuple:
                netD_out = netD_out[0]
            ldr = netD_out.detach().cpu().numpy()
        return imgs, ldr
    
    def init_drs(self):
        for i in range(50):
            _, ldr = self.get_fake_samples_and_ldr(self.batch_size)
            tmp_max = ldr.max()
            if self.maximum < tmp_max:
                self.maximum = tmp_max

    def sub_rejection_sampler(self, fake_samples, ldr, eps=1e-6):
        tmp_max = ldr.max()
        if tmp_max > self.maximum:
            self.maximum = tmp_max
        
        ldr_max = ldr - self.maximum
        
        F = ldr_max - np.log(1- np.exp(ldr_max-eps))
        if self.gamma is None:
            gamma = np.percentile(F, self.percentile)
        else:
            gamma = self.gamma

        
        F = F - gamma
        sigF = sigmoid(F)
        psi = np.random.rand(len(sigF))

        fake_x_sampled = [fake_samples[i].detach().cpu().numpy() for i in range(len(sigF)) if sigF[i] > psi[i]]
        return torch.Tensor(fake_x_sampled)
    
    def generate_images(self, num_images, device=None):
        fake_samples_list = []
        num_sampled = 0

        while num_sampled < num_images:
            fake_samples, ldrs = self.get_fake_samples_and_ldr(self.batch_size)
            fake_samples_accepted = self.sub_rejection_sampler(fake_samples, ldrs)
            fake_samples_list.append(fake_samples_accepted)
            num_sampled += fake_samples_accepted.size(0)
        fake_samples_all = torch.cat(fake_samples_list, dim=0)
        return fake_samples_all[:num_images]

