"""
PyTorch interface for computing FID.
"""
import os
import random
import time

import numpy as np
from numpy.lib.type_check import imag
import tensorflow as tf
import torch

from diagan.datasets.image_loader_with_attr import get_dataset_images_with_attr
from torch_mimicry.metrics.fid import fid_utils
from torch_mimicry.metrics.inception_model import inception_utils


def compute_real_dist_stats_with_attr(attr,
                                       sess,
                                       batch_size,
                                       dataset=None,
                                       stats_file=None,
                                       seed=0,
                                       verbose=True,
                                       log_dir='./log',
                                       name=None):
    """
    Reads the image data and compute the FID mean and cov statistics
    for real images.

    Args:
        index (ndarray): The index array of real images to use.
        sess (Session): TensorFlow session to use.
        dataset (str/Dataset): Dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        stats_file (str): The statistics file to load from if there is already one.
        verbose (bool): If True, prints progress of computation.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        ndarray: Mean features stored as np array.
        ndarray: Covariance of features stored as np array.
    """
    # Create custom stats file name
    if stats_file is None:
        stats_dir = os.path.join(log_dir, 'metrics', 'fid', 'statistics')
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        stats_file = os.path.join(
            stats_dir,
            "fid_stats_{}_{}_{}_run_{}.npz".format(name, dataset, attr,
                                                 seed))

    if stats_file and os.path.exists(stats_file):
        print("INFO: Loading existing statistics for real images...")
        f = np.load(stats_file)
        attr_m_real, attr_s_real = f['attr_mu'][:], f['attr_sigma'][:]
        not_attr_m_real, not_attr_s_real = f['not_attr_mu'][:], f['not_attr_sigma'][:]
        f.close()

    else:
        # Obtain the numpy format data
        print("INFO: Obtaining images...")
        attr_images, not_attr_images = get_dataset_images_with_attr(dataset, attr=attr)

        # Compute the mean and cov
        print("INFO: Computing statistics for real images with attribute...")
        attr_m_real, attr_s_real = fid_utils.calculate_activation_statistics(
            images=attr_images, sess=sess, batch_size=batch_size, verbose=verbose)
        not_attr_m_real, not_attr_s_real = fid_utils.calculate_activation_statistics(
            images=not_attr_images, sess=sess, batch_size=batch_size, verbose=verbose)

        if not os.path.exists(stats_file):
            print("INFO: Saving statistics for real images...")
            np.savez(stats_file, attr_mu=attr_m_real, attr_sigma=attr_s_real,
                        not_attr_mu=not_attr_m_real, not_attr_sigma=not_attr_s_real)

    return attr_m_real, attr_s_real, not_attr_m_real, not_attr_s_real


def _normalize_images(images):
    """
    Given a tensor of images, uses the torchvision
    normalization method to convert floating point data to integers. See reference
    at: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image

    The function uses the normalization from make_grid and save_image functions.

    Args:
        images (Tensor): Batch of images of shape (N, 3, H, W).

    Returns:
        ndarray: Batch of normalized images of shape (N, H, W, 3).
    """
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    images = images.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(
        'cpu', torch.uint8).numpy()

    return images


def compute_gen_dist_stats(netG,
                           num_samples,
                           sess,
                           device,
                           seed,
                           batch_size,
                           print_every=20,
                           verbose=True):
    """
    Directly produces the images and convert them into numpy format without
    saving the images on disk.

    Args:
        netG (Module): Torch Module object representing the generator model.
        num_samples (int): The number of fake images for computing statistics.
        sess (Session): TensorFlow session to use.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        batch_size (int): The number of samples per batch for inference.
        print_every (int): Interval for printing log.
        verbose (bool): If True, prints progress.

    Returns:
        ndarray: Mean features stored as np array.
        ndarray: Covariance of features stored as np array.
    """
    with torch.no_grad():
        # Set model to evaluation mode
        netG.eval()

        # Inference variables
        batch_size = min(num_samples, batch_size)

        # Collect all samples()
        images = []
        start_time = time.time()
        for idx in range(num_samples // batch_size):
            # Collect fake image
            fake_images = netG.generate_images(num_images=batch_size,
                                               device=device).detach().cpu()
            images.append(fake_images)

            # Print some statistics
            if (idx + 1) % print_every == 0:
                end_time = time.time()
                print(
                    "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                    .format(
                        (idx + 1) * batch_size, num_samples, seed,
                        (end_time - start_time) / (print_every * batch_size)))
                start_time = end_time

        # Produce images in the required (N, H, W, 3) format for FID computation
        images = torch.cat(images, 0)  # Gives (N, 3, H, W)
        images = _normalize_images(images)  # Gives (N, H, W, 3)

    # Compute the FID
    print("INFO: Computing statistics for fake images...")
    m_fake, s_fake = fid_utils.calculate_activation_statistics(
        images=images, sess=sess, batch_size=batch_size, verbose=verbose)

    return m_fake, s_fake


def fid_score_with_attr(attr,
                         num_fake_samples,
                         netG,
                         dataset,
                         seed=0,
                         device=None,
                         batch_size=50,
                         verbose=True,
                         stats_file=None,
                         log_dir='./log',
                         **kwargs):
    """
    Computes FID stats using functions that store images in memory for speed and fidelity.
    Fidelity since by storing images in memory, we don't subject the scores to different read/write
    implementations of imaging libraries.

    Args:
        attr (str): The attribute name.
        num_fake_samples (int): The number of fake images to use for FID.
        netG (Module): Torch Module object representing the generator model.
        device (str/torch.device): Device identifier to use for computation.
        seed (int): The random seed to use.
        dataset (str/Dataset): The name of the dataset to load if known, or a custom Dataset object
        batch_size (int): The batch size to feedforward for inference.
        verbose (bool): If True, prints progress.
        stats_file (str): The statistics file to load from if there is already one.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        float: Scalar FID score.
    """
    start_time = time.time()

    # Check inputs
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if isinstance(dataset, str):
        default_datasets = {
            'cifar10',
            'cifar100',
            'stl10_48',
            'imagenet_32',
            'imagenet_128',
            'celeba_64',
            'celeba_128',
            'lsun_bedroom',
            'fake_data',
        }
        if dataset not in default_datasets:
            raise ValueError('For default datasets, must be one of {}'.format(
                default_datasets))

    elif issubclass(type(dataset), torch.utils.data.Dataset):
        if stats_file is None:
            raise ValueError(
                "stats_file cannot be empty if using a custom dataset.")

        if not stats_file.endswith('.npz'):
            stats_file = stats_file + '.npz'

    else:
        raise ValueError(
            'dataset must be either a Dataset object or a string.')

    # Make sure the random seeds are fixed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Setup directories
    inception_path = os.path.join(log_dir, 'metrics', 'inception_model')

    # Setup the inception graph
    inception_utils.create_inception_graph(inception_path)

    # Start producing statistics for real and fake images
    # if device and device.index is not None:
    #     # Avoid unbounded memory usage
    #     gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,
    #                                 per_process_gpu_memory_fraction=0.15,
    #                                 visible_device_list=str(device.index))
    #     config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    # else:
    #     config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        attr_m_real, attr_s_real, not_attr_m_real, not_attr_s_real = compute_real_dist_stats_with_attr(
                                                            attr=attr,
                                                            sess=sess,
                                                            dataset=dataset,
                                                            batch_size=batch_size,
                                                            verbose=verbose,
                                                            stats_file=stats_file,
                                                            log_dir=log_dir,
                                                            seed=seed)

        m_fake, s_fake = compute_gen_dist_stats(netG=netG,
                                                num_samples=num_fake_samples,
                                                sess=sess,
                                                device=device,
                                                seed=seed,
                                                batch_size=batch_size,
                                                verbose=verbose)

        attr_FID_score = fid_utils.calculate_frechet_distance(mu1=attr_m_real,
                                                         sigma1=attr_s_real,
                                                         mu2=m_fake,
                                                         sigma2=s_fake)

        not_attr_FID_score = fid_utils.calculate_frechet_distance(mu1=not_attr_m_real,
                                                                sigma1=not_attr_s_real,
                                                                mu2=m_fake,
                                                                sigma2=s_fake)

        print("INFO: FID with attribute: {} [Time Taken: {:.4f} secs]".format(
            attr_FID_score,
            time.time() - start_time))
        print("INFO: FID with not attribute: {} [Time Taken: {:.4f} secs]".format(
            not_attr_FID_score,
            time.time() - start_time))

        return float(attr_FID_score), float(not_attr_FID_score)
