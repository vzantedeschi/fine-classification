import numpy as np
import torch

from src.datasets import CumuloDataset

path = "./datasets/cumulo-dc/"

TREE_DEPTH = 6
LR = 0.1
ITER = 2e4

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = CumuloDataset(path, ext="npz")
filename, radiances, properties, rois, labels = dataset[0]

print(radiances.shape, properties.shape, rois.shape, labels.shape)