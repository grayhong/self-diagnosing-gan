"""
PyTorch interface for computing PR.
"""
import os
import random
import time

import numpy as np
import tensorflow as tf
import torch

from diagan.datasets.image_loader_with_attr import get_dataset_images_with_attr
from torch_mimicry.metrics.inception_model import inception_utils

from diagan.trainer.compute_pr import compute_partial_recall



def compute_real_features_with_attr(attr,
                          num_samples,
                          sess,
                          batch_size,
                          dataset=None,
                          feat_file=None,
                          seed=0,
                          verbose=True,
                          log_dir='./log',
                          root='./dataset'):
    """
    Reads the image data and compute the features for real images.

    Args:
        num_samples (int): Number of real images to compute statistics.
        sess (Session): TensorFlow session to use.
        dataset (str/Dataset): Dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        feat_file (str): The features file to load from if there is already one.
        verbose (bool): If True, prints progress of computation.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        ndarray: Real data features.
    """
    # Create custom feature file name
    if feat_file is None:
        feat_dir = os.path.join(log_dir, 'metrics', 'features')
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        feat_file = os.path.join(
            feat_dir,
            "pr_feat_{}_{}_{}k_run_{}.npy".format(dataset, attr, num_samples // 1000,
                                                 seed))

    if feat_file and os.path.exists(feat_file):
        print("INFO: Loading existing features for real images...")
        f = np.load(feat_file)
        attr_real_features, not_attr_real_features = f['attr'][:], f['not_attr'][:]
        f.close()

    else:
        # Obtain the numpy format data
        print("INFO: Obtaining images...")
        attr_images, not_attr_images = get_dataset_images_with_attr(dataset, attr=attr, num_samples=num_samples, root=root)
        if len(attr_images) != num_samples:
            feat_file = os.path.join(
                feat_dir,
                "pr_feat_{}_{}_{}k_run_{}.npy".format(dataset, attr, len(attr_images) // 1000,
                                                    seed))

        # Compute the features
        print("INFO: Computing features for real images...")
        attr_real_features = inception_utils.get_activations(
            images=attr_images, sess=sess, batch_size=batch_size, verbose=verbose)
        not_attr_real_features = inception_utils.get_activations(
            images=not_attr_images, sess=sess, batch_size=batch_size, verbose=verbose)

        if not os.path.exists(feat_file):
            print("INFO: Saving features for real images...")
            np.savez(feat_file, attr=attr_real_features, not_attr=not_attr_real_features)

    return attr_real_features, not_attr_real_features


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


def compute_fake_features(netG,
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
        ndarray: Fake data features.
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
        
    # Compute the fake features
    fake_features = inception_utils.get_activations(
            images=images, sess=sess, batch_size=batch_size, verbose=verbose)

    return fake_features




def partial_recall_score_with_attr(attr,
                       num_real_samples,
                       num_fake_samples,
                       netG,
                       dataset,
                       nearest_k=3,
                       seed=0,
                       device=None,
                       batch_size=50,
                       verbose=True,
                       feat_file=None,
                       log_dir='./log'):
    """
    Computes precision and recall score with attribute.

    Args:
        attr (str): The attribute name.
        num_real_samples (int): The number of real images to use for PR.
        num_fake_samples (int): The number of fake images to use for PR.
        netG (Module): Torch Module object representing the generator model.
        device (str/torch.device): Device identifier to use for computation.
        seed (int): The random seed to use.
        dataset (str/Dataset): The name of the dataset to load if known, or a custom Dataset object
        batch_size (int): The batch size to feedforward for inference.
        nearest_k (int): The number of nearest neighborhood to use for PR.
        verbose (bool): If True, prints progress.
        feat_file (str): The features file to load from if there is already one.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        dictionary: precision, recall score dictionary.
    """
    start_time = time.time()

    # Check inputs
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if isinstance(dataset, str):
        default_datasets = {
            'celeba_64',
            'celeba_128',
            'fake_data',
        }
        if dataset not in default_datasets:
            raise ValueError('For default datasets, must be one of {}'.format(
                default_datasets))
    
    elif issubclass(type(dataset), torch.utils.data.Dataset):
        if feat_file is None:
            raise ValueError(
                "feat_file cannot be empty if using a custom dataset.")

        if not feat_file.endswith('.npy'):
            feat_file = feat_file + '.npy'

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

        attr_real_features, not_attr_real_features = compute_real_features_with_attr(attr=attr,
                                                        num_samples=num_real_samples,
                                                        sess=sess,
                                                        dataset=dataset,
                                                        batch_size=batch_size,
                                                        verbose=verbose,
                                                        feat_file=feat_file,
                                                        log_dir=log_dir,
                                                        seed=seed)


        fake_features = compute_fake_features(netG=netG,
                                              num_samples=num_fake_samples,
                                              sess=sess,
                                              device=device,
                                              seed=seed,
                                              batch_size=batch_size,
                                              verbose=verbose)


        attr_metrics = compute_partial_recall(partial_real_features=attr_real_features,
                             fake_features=fake_features,
                             nearest_k=nearest_k,
                             device=device)

        not_attr_metrics = compute_partial_recall(partial_real_features=not_attr_real_features,
                             fake_features=fake_features,
                             nearest_k=nearest_k,
                             device=device)            

        for key in attr_metrics:
            print("INFO (with attr): {}: {} [Time Taken: {:.4f} secs]".format(
                key,
                attr_metrics[key],
                time.time() - start_time))

        for key in not_attr_metrics:
            print("INFO (without attr): {}: {} [Time Taken: {:.4f} secs]".format(
                key,
                not_attr_metrics[key],
                time.time() - start_time))

        return attr_metrics, not_attr_metrics