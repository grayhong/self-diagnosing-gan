import numpy as np
from skimage.filters import threshold_otsu

def get_index_group(score):
    sorted_value = np.sort(score)
    sorted_index = np.argsort(score)
    threshold = threshold_otsu(sorted_value.reshape(-1, 1))
    index_group = dict()
    index_group[0] = sorted_index[np.where(sorted_value >= threshold)]
    index_group[1] = sorted_index[np.where(sorted_value < threshold)]
    print(f"group 0: {len(index_group[0])} 1: {len(index_group[1])}")
    return index_group