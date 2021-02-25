import torch
from torch.utils import data

import numpy as np


def generate_25_gaussians_dataset(n_base=10000, seed=1):
	np.random.seed(seed)
	rng = np.random.RandomState(seed)
	dataset = []
	labels = []
	for i in range(int((n_base) / 25)):
		for x in range(-2, 3):
			for y in range(-2, 3):
				point = rng.randn(2) * 0.05
				point[0] += 2 * x
				point[1] += 2 * y
				dataset.append(point)
				labels.append(5*(x+2)+(y+2))
	dataset = np.array(dataset, dtype='float32')
	labels = np.array(labels)
	data_labels = np.concatenate((dataset, labels[:,None]),axis =1)
	rng.shuffle(data_labels)
	dataset = data_labels[:,:2]
	labels = data_labels[:,-1]
	dataset /= 2.828  # stdev

	full_dataset = (np.array(dataset, dtype='float32'), np.array(labels))

	xs = torch.FloatTensor(full_dataset[0])
	ys = torch.LongTensor(full_dataset[1])
	base_dataset = data.TensorDataset(xs, ys)

	return base_dataset


def get_25gaussian_dataset(n_samples=10000):
    dataset = generate_25_gaussians_dataset(n_base=n_samples)
    return dataset