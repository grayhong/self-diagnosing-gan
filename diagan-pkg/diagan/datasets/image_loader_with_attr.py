import os
from imageio.core.functions import imwrite

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from torch_mimicry.datasets import data_utils
from diagan.datasets.image_loader_with_index import get_index_images
from diagan.datasets.get_celeba_index_with_attr import get_celeba_index_with_attr



def get_celeba_images_with_attr(attr, root='./dataset', size=128, num_samples=None, **kwargs):
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

    dataset_dir = os.path.join(root, 'celeba')
    attr_index, not_attr_index = get_celeba_index_with_attr(dataset_dir, attr)
    print(f'Number of images with attribute {len(attr_index)}')
    print(f'Number of images without attribute {len(not_attr_index)}')
    if num_samples:
        if len(attr_index) > num_samples:
            attr_index = np.random.choice(attr_index,
                                        size=num_samples,
                                        replace=False)
        if len(not_attr_index) > num_samples:
            not_attr_index = np.random.choice(not_attr_index,
                                        size=num_samples,
                                        replace=False)

    attr_images = get_index_images(dataset, attr_index)
    not_attr_images = get_index_images(dataset, not_attr_index)
    print(f'Number of images with attribute {attr}: {len(attr_images)}')
    print(f'Number of images without attribute {attr}: {len(not_attr_images)}')

    return attr_images, not_attr_images



def get_dataset_images_with_attr(dataset, attr, num_samples=None, **kwargs):
    if isinstance(dataset, str):
        if dataset == "celeba_64":
            attr_images, not_attr_images = get_celeba_images_with_attr(attr, size=64, num_samples=num_samples, **kwargs)
        
        elif dataset == "celeba_128":
            attr_images, not_attr_images = get_celeba_images_with_attr(attr, size=128, num_samples=num_samples, **kwargs)

        else:
            raise ValueError("Invalid dataset name {}.".format(dataset))
        
    elif issubclass(type(dataset), torch.utils.data.Dataset):
        raise NotImplementedError

    else:
        raise ValueError("dataset must be of type str or a Dataset object.")


    # Check shape and permute if needed
    if attr_images.shape[1] == 3:
        attr_images = attr_images.transpose((0, 2, 3, 1))
    if not_attr_images.shape[1] == 3:
        not_attr_images = not_attr_images.transpose((0, 2, 3, 1))

    # Ensure the values lie within the correct range, otherwise there might be some
    # preprocessing error from the library causing ill-valued scores.
    if np.min(attr_images) < 0 or np.max(attr_images) > 255:
        print(
            "INFO: Some pixel values lie outside of [0, 255]. Clipping values.."
        )
        attr_images = np.clip(attr_images, 0, 255)
    if np.min(not_attr_images) < 0 or np.max(not_attr_images) > 255:
        print(
            "INFO: Some pixel values lie outside of [0, 255]. Clipping values.."
        )
        not_attr_images = np.clip(not_attr_images, 0, 255)

    return attr_images, not_attr_images


def get_celeba_with_attr(attr, root='./dataset', split='train', size=128, **kwargs):
    """
    Loads CelebA with attribute target.

    Args:
        attr (str): The name of attribute.
        root (str): The root directory where all datasets are stored.
        split (str): The split of data to use.
        size (int): Size of image to resize to.

    Returns:
        ndarray: Batch of len(index) images in np array form.
    """
    celeba_attr = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3,
                    'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 
                    'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 
                    'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17,
                    'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21,
                    'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26,
                    'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30,
                    'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 
                    'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 
                    'Wearing_Necktie': 38, 'Young': 39}

    dataset = data_utils.load_celeba_dataset(
        root=root,
        split=split,
        size=size,
        transform_data=True,
        convert_tensor=True,
        target_transform=transforms.Lambda(lambda a: 1 if a[celeba_attr[attr]] == 1 else 0),
        **kwargs)

    return dataset