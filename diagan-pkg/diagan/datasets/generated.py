import pickle
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from diagan.datasets.transform import get_transform


class GeneratedDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.transform = transform
        self.imgs = pickle.load(open(root, 'rb'))
        self.mode = mode

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        img = Image.fromarray(img.astype(np.uint8), mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)

        return img, idx
    
    def __len__(self):
        return len(self.imgs)


def get_generated_dataset(dataset_name, root):
    transform = get_transform(dataset_name)
    if dataset_name in ['mnist_fmnist']:
        mode = 'L'
    else:
        mode = 'RGB'
    dataset = GeneratedDataset(root=root, transform=transform, mode=mode)
    return dataset