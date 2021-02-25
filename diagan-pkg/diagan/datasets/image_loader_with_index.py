"""
Loads randomly sampled images from datasets for computing metrics.
"""
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from torch_mimicry.datasets import data_utils


def get_index_images(dataset, index):
    """
    Sample with index of images.

    Args:
        dataset (Dataset): Torch Dataset object for indexing elements.
        index (ndarray): The index of images.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    choices = index

    images = []
    for choice in choices:
        img = np.array(dataset[choice][0])
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.concatenate(images, axis=0)

    return images


def get_imagenet_images_with_index(index, root='./dataset', size=32):
    """
    Directly reads the imagenet folder for obtaining images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.
        size (int): Size of image to resize to.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    if len(index) < 1000:
        raise ValueError(
            "length of index {} must be at least 1000 to ensure images are sampled from each class."
            .format(len(index)))

    data_dir = os.path.join(root, 'imagenet', 'train')
    class_dirs = os.listdir(data_dir)
    images = []

    # Randomly choose equal proportion from each class.
    for class_dir in class_dirs:
        filenames = [
            os.path.join(data_dir, class_dir, name)
            for name in os.listdir(os.path.join(data_dir, class_dir))
        ]

        choices = index
        for i in range(len(choices)):
            img = Image.open(filenames[int(choices[i])])
            img = transforms.CenterCrop(224)(img)
            img = transforms.Resize(size)(img)
            img = np.asarray(img)

            # Convert grayscale to rgb
            if len(img.shape) == 2:
                tmp = np.expand_dims(img, axis=2)
                img = np.concatenate([tmp, tmp, tmp], axis=2)

            # account for rgba images
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            img = np.expand_dims(img, axis=0)
            images.append(img)

    images = np.concatenate(images, axis=0)
    return images



def get_lsun_bedroom_images_with_index(index,
                                       root='./dataset',
                                       size=128,
                                       **kwargs):
    """
    Loads sampled LSUN-Bedroom training images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.
        size (int): Size of image to resize to.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    dataset = data_utils.load_lsun_bedroom_dataset(
        root=root,
        size=size,
        transform_data=True,
        convert_tensor=False,  # Prevents normalization.
        **kwargs)

    images = get_index_images(dataset, index)

    return images


def get_celeba_images_with_index(index, root='./dataset', size=128, **kwargs):
    """
    Loads sampled CelebA images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.
        size (int): Size of image to resize to.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    dataset = data_utils.load_celeba_dataset(
        root=root,
        size=size,
        transform_data=True,
        convert_tensor=False,  # Prevents normalization.
        **kwargs)

    images = get_index_images(dataset, index)

    return images


def get_stl10_images_with_index(index, root='./dataset', size=48, **kwargs):
    """
    Loads sampled STL-10 images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.
        size (int): Size of image to resize to.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    dataset = data_utils.load_stl10_dataset(
        root=root,
        size=size,
        transform_data=True,
        convert_tensor=False,  # Prevents normalization.
        **kwargs)

    images = get_index_images(dataset, index)

    return images


def get_cifar10_images_with_index(index, root="./dataset", **kwargs):
    """
    Loads sampled CIFAR-10 training images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    dataset = data_utils.load_cifar10_dataset(root=root,
                                              transform_data=False,
                                              **kwargs)

    images = get_index_images(dataset, index)

    return images


def get_cifar100_images_with_index(index, root="./dataset", **kwargs):
    """
    Loads sampled CIFAR-100 training images with index.

    Args:
        index (ndarray): The index of images.
        root (str): The root directory where all datasets are stored.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    dataset = data_utils.load_cifar100_dataset(root=root,
                                               split='train',
                                               download=True,
                                               transform_data=False,
                                               **kwargs)

    images = get_index_images(dataset, index)

    return images


def sample_dataset_images_with_index(dataset, index):
    """
    Samples the dataset for images with index.

    Args:
        dataset (Dataset): Torch dataset object to sample images from.
        index (ndarray): The index of images.

    Returns:
        ndarray: Numpy array of images with first dim as batch size.
    """
    # Check if sufficient images
    if len(dataset) < len(index):
        raise ValueError(
            "Given dataset has less than len(index) images: {} given but requires at least {}."
            .format(len(dataset), len(index)))

    choices = index
    images = []
    for i in choices:
        data = dataset[i]

        # Case of iterable, assumes first arg is data.
        if isinstance(data, tuple) or isinstance(data, list):
            img = data[0]
        else:
            img = data

        img = torch.unsqueeze(img, 0)
        images.append(img)

    images = np.concatenate(images, axis=0)

    return images


def get_dataset_images_with_index(dataset, index, **kwargs):
    """
    Sample images with index based on input dataset name.

    Args:
        dataset (str/Dataset): Dataset to load images from.
        index (ndarray): The index of images.

    Returns:
        ndarray: Batch of len(index) images from a dataset in np array form.
            The final format is of (N, H, W, 3) shape for TF inference.
    """
    if isinstance(dataset, str):
        if dataset == "imagenet_32":
            images = get_imagenet_images_with_index(index, size=32, **kwargs)

        elif dataset == "imagenet_128":
            images = get_imagenet_images_with_index(index, size=128, **kwargs)

        elif dataset == "celeba_64":
            images = get_celeba_images_with_index(index, size=64, **kwargs)

        elif dataset == "celeba_128":
            images = get_celeba_images_with_index(index, size=128, **kwargs)

        elif dataset == "stl10_48":
            images = get_stl10_images_with_index(index, **kwargs)

        elif dataset == "cifar10":
            images = get_cifar10_images_with_index(index, **kwargs)

        elif dataset == "cifar100":
            images = get_cifar100_images_with_index(index, **kwargs)

        elif dataset == "lsun_bedroom_128":
            images = get_lsun_bedroom_images_with_index(index, size=128, **kwargs)

        else:
            raise ValueError("Invalid dataset name {}.".format(dataset))

    elif issubclass(type(dataset), torch.utils.data.Dataset):
        images = sample_dataset_images_with_index(dataset, index)

    else:
        raise ValueError("dataset must be of type str or a Dataset object.")

    # Check shape and permute if needed
    if images.shape[1] == 3:
        images = images.transpose((0, 2, 3, 1))

    # Ensure the values lie within the correct range, otherwise there might be some
    # preprocessing error from the library causing ill-valued scores.
    if np.min(images) < 0 or np.max(images) > 255:
        print(
            "INFO: Some pixel values lie outside of [0, 255]. Clipping values.."
        )
        images = np.clip(images, 0, 255)

    return images
