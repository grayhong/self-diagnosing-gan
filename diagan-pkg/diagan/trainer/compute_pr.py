"""
Modified from https://github.com/clovaai/generative-evaluation-prdc
"""
import numpy as np
import torch
import sklearn.metrics

__all__ = ['compute_pr']


def compute_pairwise_distance(data_x, data_y=None, device=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    # dists = sklearn.metrics.pairwise_distances(
    #     data_x, data_y, metric='euclidean', n_jobs=8)
    data_x = torch.from_numpy(data_x).to(device)
    data_y = torch.from_numpy(data_y).to(device)

    norm_x = torch.sum(torch.square(data_x), dim=1).unsqueeze_(-1)
    norm_y = torch.sum(torch.square(data_y), dim=1).unsqueeze_(0)

    dists = norm_x - 2 * torch.matmul(data_x, torch.transpose(data_y, 1, 0)) + norm_y

    return dists.cpu().numpy()


def get_kth_value(unsorted, k, axis=-1, device=None):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    # indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    # k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    # kth_values = k_smallests.max(axis=axis)
    unsorted = torch.from_numpy(unsorted).to(device)
    _, indices = torch.topk(unsorted, k, largest=False)
    k_smallests = torch.gather(unsorted, axis, indices)
    kth_values, _ = torch.max(k_smallests, axis)
    return kth_values.cpu().numpy()


def compute_nearest_neighbour_distances(input_features, nearest_k, device=None):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features, device=device)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1, device=device)
    return radii


def compute_pr(real_features, fake_features, nearest_k, device=None):
    """
    Computes precision and recall given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k, device=device)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k, device=device)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features, device=device)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    return dict(precision=precision, recall=recall)


def compute_partial_recall(partial_real_features, fake_features, nearest_k, device=None):
    """
    Computes partial recall given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of partial recall.
    """

    print('Num real: {} Num fake: {}'
          .format(partial_real_features.shape[0], fake_features.shape[0]))

    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k, device=device)
    distance_real_fake = compute_pairwise_distance(
        partial_real_features, fake_features, device=device)

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    return dict(recall=recall)
