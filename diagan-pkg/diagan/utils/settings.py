import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

def set_seed(seed=3):
    if seed is not None:
        print(f'=======> Using Fixed Random Seed: {seed} <========')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = True # set to False for final report
