import cv2
import numpy as np 

def remove_outliers(data, thresh=1.5, axis=(0,1), use_median=False):
    # Post: Remove outlier values from data. A value in data is considered an outlier if it is NOT mean-std_deviation*thresh < value < mean+std_deviation*thresh

    res                 = []
    median              = np.median(data, axis) 
    mean, std_dev       = cv2.meanStdDev(data)
    measure             = median if use_median else mean
    lower_thresh        = np.subtract(measure, np.multiply(std_dev, thresh))
    upper_thresh        = np.add(measure, np.multiply(std_dev, thresh))

    # Handle arrays that are n dimensional
    if len(data.shape) == 3:
        data = data.reshape((int(data.size/3), 3))

    for v in data:
        if np.all(v > lower_thresh) and np.all(v < upper_thresh):
            res.append(v)

    return res