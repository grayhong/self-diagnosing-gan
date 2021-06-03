"""
Computes different GAN metrics for a generator.
"""
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from torch_mimicry.metrics import compute_fid, compute_is, compute_kid
from torch_mimicry.utils import common

from torchvision import utils as vutils

from .distributed import *
from .pr_score import pr_score
from .fid_score import fid_score
from .compute_fid_with_index import fid_score_with_index
from .compute_fid_with_attr import fid_score_with_attr
from .pr_score_with_attr import partial_recall_score_with_attr

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DRS(nn.Module):
    def __init__(self, netG, netD, device, batch_size=256):
        super().__init__()
        self.netG = netG
        self.netD = netD
        self.maximum = -100000
        self.device = device
        self.batch_size = batch_size
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

    def sub_rejection_sampler(self, fake_samples, ldr, eps=1e-6, gamma=None):
        tmp_max = ldr.max()
        if tmp_max > self.maximum:
            self.maximum = tmp_max
        
        ldr_max = ldr - self.maximum
        
        F = ldr_max - np.log(1- np.exp(ldr_max-eps))
        if gamma is None:
            gamma = np.percentile(F, 80)
        
        F = F - gamma
        sigF = sigmoid(F)
        psi = np.random.rand(len(sigF))

        fake_x_sampled = [fake_samples[i].detach().cpu().numpy() for i in range(len(sigF)) if sigF[i] > psi[i]]
        return torch.Tensor(fake_x_sampled)
    
    def generate_images(self, num_images, device=None):
        fake_samples_list = []
        num_sampled = 0

        if device is None:
            device = self.device

        while num_sampled < num_images:
            fake_samples, ldrs = self.get_fake_samples_and_ldr(self.batch_size)
            fake_samples_accepted = self.sub_rejection_sampler(fake_samples, ldrs)
            fake_samples_list.append(fake_samples_accepted)
            num_sampled += fake_samples_accepted.size(0)
        fake_samples_all = torch.cat(fake_samples_list, dim=0)
        return fake_samples_all[:num_images].to(device)
    
    def visualize_images(self, log_dir, evaluate_step, num_images = 64):
        img_dir = os.path.join(log_dir, 'images')
        fake_samples = self.generate_images(num_images)

        images_viz = vutils.make_grid(fake_samples,
                                        padding=2,
                                        normalize=True)

        vutils.save_image(images_viz,
                '{}/fake_samples_step_{}_after_drs.png'.format(img_dir, evaluate_step),
                normalize=True)

def evaluate_drs(
    metric,
    netG,
    netD_drs,
    log_dir,
    evaluate_range=None,
    evaluate_step=None,
    use_original_netD=False,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    is_stylegan2=False,
    **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'kid':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for KID computation.")

        output_file = os.path.join(
            output_log_dir, 'kid_{}k.json'.format(kwargs['num_samples'] // 1000))

    elif metric == 'fid':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                      kwargs['num_fake_samples'] // 1000))

    elif metric == 'inception_score':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for IS computation.")

        output_file = os.path.join(
            output_log_dir,
            'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    elif metric == 'pr':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for PR computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'pr_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                       kwargs['num_fake_samples'] // 1000))

    else:
        choices = ['fid', 'kid', 'inception_score', 'pr']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    netG_ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if use_original_netD:
        ckpt_path = 'netD'
    else:
        ckpt_path = 'netD_drs'
    netD_drs_ckpt_dir = os.path.join(log_dir, 'checkpoints', ckpt_path)
    if not os.path.exists(netG_ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                netG_ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
        'inception_score': 'Inception Score',
        'kid': 'KID',
        'pr': 'PR',
    }

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        # if step in scores_dict and write_to_json and not overwrite:
        #     print("INFO: {} at step {} has been computed. Skipping...".format(
        #         names_dict[metric], step))
        #     continue

        # Load and restore the model checkpoint
        netG_ckpt_file = os.path.join(netG_ckpt_dir, 'netG_{}_steps.pth'.format(step))
        netD_drs_ckpt_file = os.path.join(netD_drs_ckpt_dir, f'{ckpt_path}_{step}_steps.pth')
        if not os.path.exists(netG_ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=netG_ckpt_file, optimizer=None)
        if is_stylegan2:
            ckpt = torch.load(netG_ckpt_file, map_location=lambda storage, loc: storage)
            netD_drs.load_state_dict(ckpt["drs_d"] if "drs_d" in ckpt else ckpt["d"])
        else:
            netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt_file, optimizer=None)

        netG = DRS(netG=netG, netD=netD_drs, device=device)

        #Visualize images after DRS
        netG.visualize_images(log_dir = log_dir, evaluate_step = evaluate_step)

        # Compute score for each seed
        scores = []
        if metric == 'pr':
            scores = defaultdict(list)
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = compute_fid.fid_score(netG=netG,
                                              seed=seed,
                                              device=device,
                                              log_dir=log_dir,
                                              **kwargs)

            elif metric == "inception_score":
                score, _ = compute_is.inception_score(netG=netG,
                                                      seed=seed,
                                                      device=device,
                                                      log_dir=log_dir,
                                                      **kwargs)

            elif metric == "kid":
                score, _ = compute_kid.kid_score(netG=netG,
                                                 device=device,
                                                 seed=seed,
                                                 log_dir=log_dir,
                                                 **kwargs)

            elif metric == "pr":
                score = pr_score(netG=netG,
                                 seed=seed,
                                 device=device,
                                 log_dir=log_dir,
                                 **kwargs)

            if metric == "pr":
                for key in score:
                    scores[key].append(score[key])
                    print("INFO: {} (step {}) [seed {}]: {}".format(
                        key, step, seed, score[key]))
            else:
                scores.append(score)
                print("INFO: {} (step {}) [seed {}]: {}".format(
                    names_dict[metric], step, seed, score))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            if metric == "pr":
                for key in scores_dict[step]:
                    scores = scores_dict[step][key]
                    mean = np.mean(scores)
                    std = np.std(scores)

                    print("INFO: {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))
            else:
                scores = scores_dict[step]
                mean = np.mean(scores)
                std = np.std(scores)

                print("INFO: {} (step {}): {} (± {}) ".format(
                    names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict



def evaluate_custom(
    metric,
    netG,
    log_dir,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    **kwargs):

    output_log_dir = log_dir / 'evaluate' / f'custom'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'kid':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for KID computation.")

        output_file = os.path.join(
            output_log_dir, 'kid_{}k.json'.format(kwargs['num_samples'] // 1000))

    elif metric == 'fid':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                      kwargs['num_fake_samples'] // 1000))

    elif metric == 'inception_score':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for IS computation.")

        output_file = os.path.join(
            output_log_dir,
            'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    else:
        choices = ['fid', 'kid', 'inception_score']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
        'inception_score': 'Inception Score',
        'kid': 'KID',
    }
    
    step = 0
    # Compute score for each seed
    scores = []
    for seed in range(start_seed, start_seed + num_runs):
        print("INFO: Computing {} in memory...".format(names_dict[metric]))

        # Obtain only the raw score without var
        if metric == "fid":
            score = compute_fid.fid_score(netG=netG,
                                            seed=seed,
                                            device=device,
                                            log_dir=log_dir,
                                            **kwargs)

        elif metric == "inception_score":
            score, _ = compute_is.inception_score(netG=netG,
                                                    seed=seed,
                                                    device=device,
                                                    log_dir=log_dir,
                                                    **kwargs)

        elif metric == "kid":
            score, _ = compute_kid.kid_score(netG=netG,
                                                device=device,
                                                seed=seed,
                                                log_dir=log_dir,
                                                **kwargs)

        scores.append(score)
        print("INFO: {} (step {}) [seed {}]: {}".format(
            names_dict[metric], step, seed, score))

    scores_dict[step] = scores

    # Save scores every step
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    if step in scores_dict:
        scores = scores_dict[step]
        mean = np.mean(scores)
        std = np.std(scores)

        print("INFO: {} (step {}): {} (± {}) ".format(
            names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict

def evaluate_pr(
    netG,
    log_dir,
    evaluate_range=None,
    evaluate_step=None,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    **kwargs):
    """
    Computes precision and recall.

    Args:
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        dictionary: precision, recall score dictionary.
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
        raise ValueError(
            "num_real_samples and num_fake_samples must be provided for PR computation."
        )

    output_file = os.path.join(
        output_log_dir,
        'pr_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                    kwargs['num_fake_samples'] // 1000))

    # Check checkpoint dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        # if step in scores_dict and write_to_json and not overwrite:
        #     print("INFO: PR at step {} has been computed. Skipping...".format(step))
        #     continue

        # Load and restore the model checkpoint
        ckpt_file = os.path.join(ckpt_dir, 'netG_{}_steps.pth'.format(step))
        if not os.path.exists(ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

        # Compute score for each seed
        scores = defaultdict(list)
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing PR in memory...")

            # Obtain only the raw score without var
            score = pr_score(netG=netG,
                             seed=seed,
                             device=device,
                             log_dir=log_dir,
                             **kwargs)
            for key in score:
                scores[key].append(score[key])
                print("INFO: {} (step {}) [seed {}]: {}".format(
                    key, step, seed, score[key]))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            for key in scores_dict[step]:
                scores = scores_dict[step][key]
                mean = np.mean(scores)
                std = np.std(scores)

                print("INFO: {} (step {}): {} (± {}) ".format(
                    key, step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: PR Evaluation completed!")

    return scores_dict


def evaluate_with_index(metric,
                        index,
                        netG,
                        log_dir,
                        evaluate_range=None,
                        evaluate_step=None,
                        num_runs=3,
                        start_seed=0,
                        overwrite=False,
                        write_to_json=True,
                        device=None,
                        **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        index (ndarray): The index array of real images to use.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'fid':
        if 'name' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "name and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            log_dir,
            'fid_{}_{}_{}k.json'.format(kwargs['name'], len(index),
                                      kwargs['num_fake_samples'] // 1000))

    else:
        choices = 'fid'
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
    }

    # # Set output file and restore if available.
    # if metric == 'fid':
    #     output_file = os.path.join(
    #         log_dir,
    #         'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
    #                                   kwargs['num_fake_samples'] // 1000))

    # elif metric == 'inception_score':
    #     output_file = os.path.join(
    #         log_dir,
    #         'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    # elif metric == 'kid':
    #     output_file = os.path.join(
    #         log_dir, 'kid_{}k.json'.format(
    #             kwargs['num_samples'] // 1000))

    # if os.path.exists(output_file):
    #     scores_dict = common.load_from_json(output_file)
    #     scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    # else:
    #     scores_dict = {}

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        if step in scores_dict and write_to_json and not overwrite:
            print("INFO: {} at step {} has been computed. Skipping...".format(
                names_dict[metric], step))
            continue

        # Load and restore the model checkpoint
        ckpt_file = os.path.join(ckpt_dir, 'netG_{}_steps.pth'.format(step))
        if not os.path.exists(ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

        # Compute score for each seed
        scores = []
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = fid_score_with_index(index=index,
                                             netG=netG,
                                             seed=seed,
                                             device=device,
                                             log_dir=log_dir,
                                             **kwargs)

            scores.append(score)
            print("INFO: {} (step {}) [seed {}]: {}".format(
                names_dict[metric], step, seed, score))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            scores = scores_dict[step]
            mean = np.mean(scores)
            std = np.std(scores)

            print("INFO: {} (step {}): {} (± {}) ".format(
                names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict


def evaluate_drs_with_index(
    metric,
    index,
    netG,
    netD_drs,
    log_dir,
    evaluate_range=None,
    evaluate_step=None,
    use_original_netD=False,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        index (ndarray): The index array of real images to use.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'fid':
        if 'name' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "name and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            log_dir,
            'fid_{}_drs_{}_{}k.json'.format(kwargs['name'], len(index),
                                      kwargs['num_fake_samples'] // 1000))

    else:
        choices = 'fid'
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    netG_ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if use_original_netD:
        ckpt_path = 'netD'
    else:
        ckpt_path = 'netD_drs'
    netD_drs_ckpt_dir = os.path.join(log_dir, 'checkpoints', ckpt_path)
    if not os.path.exists(netG_ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                netG_ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
    }

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        # if step in scores_dict and write_to_json and not overwrite:
        #     print("INFO: {} at step {} has been computed. Skipping...".format(
        #         names_dict[metric], step))
        #     continue

        # Load and restore the model checkpoint
        netG_ckpt_file = os.path.join(netG_ckpt_dir, 'netG_{}_steps.pth'.format(step))
        netD_drs_ckpt_file = os.path.join(netD_drs_ckpt_dir, f'{ckpt_path}_{step}_steps.pth')
        if not os.path.exists(netG_ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=netG_ckpt_file, optimizer=None)
        netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt_file, optimizer=None)

        netG = DRS(netG=netG, netD=netD_drs, device=device)

        #Visualize images after DRS
        netG.visualize_images(log_dir = log_dir, evaluate_step = evaluate_step)

        # Compute score for each seed
        scores = []
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = fid_score_with_index(index=index,
                                             netG=netG,
                                             seed=seed,
                                             device=device,
                                             log_dir=log_dir,
                                             **kwargs)

            scores.append(score)
            print("INFO: {} (step {}) [seed {}]: {}".format(
                names_dict[metric], step, seed, score))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            scores = scores_dict[step]
            mean = np.mean(scores)
            std = np.std(scores)

            print("INFO: {} (step {}): {} (± {}) ".format(
                names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict



def evaluate_with_attr(metric,
                        attr,
                        netG,
                        log_dir,
                        evaluate_range=None,
                        evaluate_step=None,
                        num_runs=3,
                        start_seed=0,
                        overwrite=False,
                        write_to_json=True,
                        device=None,
                        **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        attr (str): The attribute name.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'partial_recall':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for PR computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'partial_recall_{}_{}k_{}k.json'.format(attr, kwargs['num_real_samples'] // 1000,
                                    kwargs['num_fake_samples'] // 1000))

    else:
        choices = ['partial_recall']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        attr_scores_dict = scores_dict['attr']
        not_attr_scores_dict = scores_dict['not_attr']
        attr_scores_dict = dict([(int(k), v) for k, v in attr_scores_dict.items()])
        not_attr_scores_dict = dict([(int(k), v) for k, v in not_attr_scores_dict.items()])

    else:
        scores_dict = {}
        attr_scores_dict = {}
        not_attr_scores_dict = {}

    # Decide naming convention
    names_dict = {
        'partial_recall': 'Partial Recall'
    }

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        if step in scores_dict and write_to_json and not overwrite:
            print("INFO: {} at step {} has been computed. Skipping...".format(
                names_dict[metric], step))
            continue

        # Load and restore the model checkpoint
        ckpt_file = os.path.join(ckpt_dir, 'netG_{}_steps.pth'.format(step))
        if not os.path.exists(ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

        # Compute score for each seed
        attr_scores = []
        not_attr_scores = []
        attr_pr_scores = defaultdict(list)
        not_attr_pr_scores = defaultdict(list)
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "partial_recall":
                attr_score, not_attr_score = partial_recall_score_with_attr(attr=attr,
                                            netG=netG,
                                            seed=seed,
                                            device=device,
                                            log_dir=log_dir,
                                            **kwargs)
                for key in attr_score:
                    attr_pr_scores[key].append(attr_score[key])
                    not_attr_pr_scores[key].append(not_attr_score[key])
                    print("INFO (with attr): {} (step {}) [seed {}]: {}".format(
                        key, step, seed, attr_score[key]))
                    print("INFO (without attr): {} (step {}) [seed {}]: {}".format(
                        key, step, seed, not_attr_score[key]))
            

        attr_scores_dict[step] = attr_scores
        not_attr_scores_dict[step] = not_attr_scores
        scores_dict = {'attr':attr_scores_dict, 'not_attr':not_attr_scores_dict}

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in attr_scores_dict:
            if metric == "partial_recall":
                for key in attr_scores_dict[step]:
                    attr_scores = attr_scores_dict[step][key]
                    mean = np.mean(attr_scores)
                    std = np.mean(attr_scores)

                    print("INFO (with attr): {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))

        
        if step in not_attr_scores_dict:
            if metric == "partial_recall":
                for key in not_attr_scores_dict[step]:
                    not_attr_scores = not_attr_scores_dict[step][key]
                    mean = np.mean(not_attr_scores)
                    std = np.mean(not_attr_scores)

                    print("INFO (without attr): {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict


def evaluate_drs_with_attr(
    metric,
    attr,
    netG,
    netD_drs,
    log_dir,
    evaluate_range=None,
    evaluate_step=None,
    use_original_netD=False,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        attr (str): The attribute name.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'partial_recall':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for PR computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'partial_recall_{}_{}k_{}k.json'.format(attr, kwargs['num_real_samples'] // 1000,
                                    kwargs['num_fake_samples'] // 1000))

    else:
        choices = ['partial_recall']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    netG_ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if use_original_netD:
        ckpt_path = 'netD'
    else:
        ckpt_path = 'netD_drs'
    netD_drs_ckpt_dir = os.path.join(log_dir, 'checkpoints', ckpt_path)
    if not os.path.exists(netG_ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                netG_ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        attr_scores_dict = scores_dict['attr']
        not_attr_scores_dict = scores_dict['not_attr']
        attr_scores_dict = dict([(int(k), v) for k, v in attr_scores_dict.items()])
        not_attr_scores_dict = dict([(int(k), v) for k, v in not_attr_scores_dict.items()])

    else:
        scores_dict= {}
        attr_scores_dict = {}
        not_attr_scores_dict = {}

    # Decide naming convention
    names_dict = {
        'partial_recall': 'Partial Recall'
    }

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        # if step in scores_dict and write_to_json and not overwrite:
        #     print("INFO: {} at step {} has been computed. Skipping...".format(
        #         names_dict[metric], step))
        #     continue

        # Load and restore the model checkpoint
        netG_ckpt_file = os.path.join(netG_ckpt_dir, 'netG_{}_steps.pth'.format(step))
        netD_drs_ckpt_file = os.path.join(netD_drs_ckpt_dir, f'{ckpt_path}_{step}_steps.pth')
        if not os.path.exists(netG_ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=netG_ckpt_file, optimizer=None)
        netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt_file, optimizer=None)

        netG = DRS(netG=netG, netD=netD_drs, device=device)

        #Visualize images after DRS
        netG.visualize_images(log_dir = log_dir, evaluate_step = evaluate_step)

        # Compute score for each seed
        attr_scores = []
        not_attr_scores = []
        attr_pr_scores = defaultdict(list)
        not_attr_pr_scores = defaultdict(list)
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "partial_recall":
                attr_score, not_attr_score = partial_recall_score_with_attr(attr=attr,
                                            netG=netG,
                                            seed=seed,
                                            device=device,
                                            log_dir=log_dir,
                                            **kwargs)
                for key in attr_score:
                    attr_pr_scores[key].append(attr_score[key])
                    not_attr_pr_scores[key].append(not_attr_score[key])
                    print("INFO (with attr): {} (step {}) [seed {}]: {}".format(
                        key, step, seed, attr_score[key]))
                    print("INFO (without attr): {} (step {}) [seed {}]: {}".format(
                        key, step, seed, not_attr_score[key]))
            

        attr_scores_dict[step] = attr_scores
        not_attr_scores_dict[step] = not_attr_scores
        scores_dict = {'attr':attr_scores_dict, 'not_attr':not_attr_scores_dict}

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in attr_scores_dict:
            if metric == "partial_recall":
                for key in attr_scores_dict[step]:
                    attr_scores = attr_scores_dict[step][key]
                    mean = np.mean(attr_scores)
                    std = np.mean(attr_scores)

                    print("INFO (with attr): {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))

        
        if step in not_attr_scores_dict:
            if metric == "partial_recall":
                for key in not_attr_scores_dict[step]:
                    not_attr_scores = not_attr_scores_dict[step][key]
                    mean = np.mean(not_attr_scores)
                    std = np.mean(not_attr_scores)

                    print("INFO (without attr): {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict


def evaluate_ffhq(metric,
                  netG,
                  log_dir,
                  evaluate_range=None,
                  evaluate_step=None,
                  num_runs=3,
                  start_seed=0,
                  overwrite=False,
                  write_to_json=True,
                  device=None,
                  **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    # Check metric arguments
    if metric == 'kid':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for KID computation.")

        output_file = os.path.join(
            log_dir, 'kid_{}k.json'.format(kwargs['num_samples'] // 1000))

    elif metric == 'fid':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            log_dir,
            'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                      kwargs['num_fake_samples'] // 1000))

    elif metric == 'inception_score':
        if 'num_samples' not in kwargs:
            raise ValueError(
                "num_samples must be provided for IS computation.")

        output_file = os.path.join(
            log_dir,
            'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    else:
        choices = ['fid', 'kid', 'inception_score']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if not os.path.exists(ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
        'inception_score': 'Inception Score',
        'kid': 'KID',
    }

    # # Set output file and restore if available.
    # if metric == 'fid':
    #     output_file = os.path.join(
    #         log_dir,
    #         'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
    #                                   kwargs['num_fake_samples'] // 1000))

    # elif metric == 'inception_score':
    #     output_file = os.path.join(
    #         log_dir,
    #         'inception_score_{}k.json'.format(kwargs['num_samples'] // 1000))

    # elif metric == 'kid':
    #     output_file = os.path.join(
    #         log_dir, 'kid_{}k.json'.format(
    #             kwargs['num_samples'] // 1000))

    # if os.path.exists(output_file):
    #     scores_dict = common.load_from_json(output_file)
    #     scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    # else:
    #     scores_dict = {}

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        if step in scores_dict and write_to_json and not overwrite:
            print("INFO: {} at step {} has been computed. Skipping...".format(
                names_dict[metric], step))
            continue

        # Load and restore the model checkpoint
        ckpt_file = os.path.join(ckpt_dir, 'netG_{}_steps.pth'.format(step))
        if not os.path.exists(ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=ckpt_file, optimizer=None)

        # Compute score for each seed
        scores = []
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = fid_score(netG=netG,
                                  seed=seed,
                                  device=device,
                                  log_dir=log_dir,
                                  **kwargs)

            elif metric == "inception_score":
                raise NotImplementedError
                # score, _ = compute_is.inception_score(netG=netG,
                #                                       seed=seed,
                #                                       device=device,
                #                                       log_dir=log_dir,
                #                                       **kwargs)

            elif metric == "kid":
                raise NotImplementedError
                # score, _ = compute_kid.kid_score(netG=netG,
                #                                  device=device,
                #                                  seed=seed,
                #                                  log_dir=log_dir,
                #                                  **kwargs)

            scores.append(score)
            print("INFO: {} (step {}) [seed {}]: {}".format(
                names_dict[metric], step, seed, score))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            scores = scores_dict[step]
            mean = np.mean(scores)
            std = np.std(scores)

            print("INFO: {} (step {}): {} (± {}) ".format(
                names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict


def evaluate_drs_ffhq(
    metric,
    netG,
    netD_drs,
    log_dir,
    evaluate_range=None,
    evaluate_step=None,
    use_original_netD=False,
    num_runs=1,
    start_seed=0,
    overwrite=False,
    write_to_json=True,
    device=None,
    is_stylegan2=False,
    **kwargs):
    """
    Evaluates a generator over several runs.

    Args:
        metric (str): The name of the metric for evaluation.
        netG (Module): Torch generator model to evaluate.
        log_dir (str): The path to the log directory.
        evaluate_range (tuple): The 3 valued tuple for defining a for loop.
        evaluate_step (int): The specific checkpoint to load. Used in place of evaluate_range.
        device (str): Device identifier to use for computation.
        num_runs (int): The number of runs to compute FID for each checkpoint.
        start_seed (int): Starting random seed to use.
        write_to_json (bool): If True, writes to an output json file in log_dir.
        overwrite (bool): If True, then overwrites previous metric score.

    Returns:
        None
    """
    # Check evaluation range/steps
    if evaluate_range and evaluate_step or not (evaluate_step
                                                or evaluate_range):
        raise ValueError(
            "Only one of evaluate_step or evaluate_range can be defined.")

    if evaluate_range:
        if (type(evaluate_range) != tuple
                or not all(map(lambda x: type(x) == int, evaluate_range))
                or not len(evaluate_range) == 3):
            raise ValueError(
                "evaluate_range must be a tuple of ints (start, end, step).")

    output_log_dir = log_dir / 'evaluate' / f'step-{evaluate_step}'
    output_log_dir.mkdir(parents=True, exist_ok=True)
    # Check metric arguments
    if metric == 'kid':
        raise NotImplementedError

    elif metric == 'fid':
        if 'num_real_samples' not in kwargs or 'num_fake_samples' not in kwargs:
            raise ValueError(
                "num_real_samples and num_fake_samples must be provided for FID computation."
            )

        output_file = os.path.join(
            output_log_dir,
            'fid_{}k_{}k.json'.format(kwargs['num_real_samples'] // 1000,
                                      kwargs['num_fake_samples'] // 1000))

    elif metric == 'inception_score':
        raise NotImplementedError

    elif metric == 'pr':
        raise NotImplementedError

    else:
        choices = ['fid', 'kid', 'inception_score', 'pr']
        raise ValueError("Invalid metric {} selected. Choose from {}.".format(
            metric, choices))

    # Check checkpoint dir
    netG_ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netG')
    if use_original_netD:
        ckpt_path = 'netD'
    else:
        ckpt_path = 'netD_drs'
    netD_drs_ckpt_dir = os.path.join(log_dir, 'checkpoints', ckpt_path)
    if not os.path.exists(netG_ckpt_dir):
        raise ValueError(
            "Checkpoint directory {} cannot be found in log_dir.".format(
                netG_ckpt_dir))

    # Check device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup output file
    if os.path.exists(output_file):
        scores_dict = common.load_from_json(output_file)
        scores_dict = dict([(int(k), v) for k, v in scores_dict.items()])

    else:
        scores_dict = {}

    # Decide naming convention
    names_dict = {
        'fid': 'FID',
        'inception_score': 'Inception Score',
        'kid': 'KID',
        'pr': 'PR',
    }

    # Evaluate across a range
    start, end, interval = evaluate_range or (evaluate_step, evaluate_step,
                                              evaluate_step)
    for step in range(start, end + 1, interval):
        # Skip computed scores
        # if step in scores_dict and write_to_json and not overwrite:
        #     print("INFO: {} at step {} has been computed. Skipping...".format(
        #         names_dict[metric], step))
        #     continue

        # Load and restore the model checkpoint
        netG_ckpt_file = os.path.join(netG_ckpt_dir, 'netG_{}_steps.pth'.format(step))
        netD_drs_ckpt_file = os.path.join(netD_drs_ckpt_dir, f'{ckpt_path}_{step}_steps.pth')
        if not os.path.exists(netG_ckpt_file):
            print("INFO: Checkpoint at step {} does not exist. Skipping...".
                  format(step))
            continue
        netG.restore_checkpoint(ckpt_file=netG_ckpt_file, optimizer=None)
        if is_stylegan2:
            ckpt = torch.load(netG_ckpt_file, map_location=lambda storage, loc: storage)
            netD_drs.load_state_dict(ckpt["drs_d"] if "drs_d" in ckpt else ckpt["d"])
        else:
            netD_drs.restore_checkpoint(ckpt_file=netD_drs_ckpt_file, optimizer=None)

        netG = DRS(netG=netG, netD=netD_drs, device=device, batch_size=128)

        #Visualize images after DRS
        netG.visualize_images(log_dir = log_dir, evaluate_step = evaluate_step)

        # Compute score for each seed
        scores = []
        if metric == 'pr':
            scores = defaultdict(list)
        for seed in range(start_seed, start_seed + num_runs):
            print("INFO: Computing {} in memory...".format(names_dict[metric]))

            # Obtain only the raw score without var
            if metric == "fid":
                score = fid_score(netG=netG,
                                              seed=seed,
                                              device=device,
                                              log_dir=log_dir,
                                            #   batch_size=10,
                                              **kwargs)

            elif metric == "inception_score":
                score, _ = compute_is.inception_score(netG=netG,
                                                      seed=seed,
                                                      device=device,
                                                      log_dir=log_dir,
                                                      **kwargs)

            elif metric == "kid":
                score, _ = compute_kid.kid_score(netG=netG,
                                                 device=device,
                                                 seed=seed,
                                                 log_dir=log_dir,
                                                 **kwargs)

            elif metric == "pr":
                score = pr_score(netG=netG,
                                 seed=seed,
                                 device=device,
                                 log_dir=log_dir,
                                 **kwargs)

            if metric == "pr":
                for key in score:
                    scores[key].append(score[key])
                    print("INFO: {} (step {}) [seed {}]: {}".format(
                        key, step, seed, score[key]))
            else:
                scores.append(score)
                print("INFO: {} (step {}) [seed {}]: {}".format(
                    names_dict[metric], step, seed, score))

        scores_dict[step] = scores

        # Save scores every step
        if write_to_json:
            common.write_to_json(scores_dict, output_file)

    # Print the scores in order
    for step in range(start, end + 1, interval):
        if step in scores_dict:
            if metric == "pr":
                for key in scores_dict[step]:
                    scores = scores_dict[step][key]
                    mean = np.mean(scores)
                    std = np.std(scores)

                    print("INFO: {} (step {}): {} (± {}) ".format(
                        key, step, mean, std))
            else:
                scores = scores_dict[step]
                mean = np.mean(scores)
                std = np.std(scores)

                print("INFO: {} (step {}): {} (± {}) ".format(
                    names_dict[metric], step, mean, std))

    # Save to output file
    if write_to_json:
        common.write_to_json(scores_dict, output_file)

    print("INFO: {} Evaluation completed!".format(names_dict[metric]))

    return scores_dict