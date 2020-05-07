import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import CumuloDataset, class_pixel_collate
from src.optimization import train_stochastic

path = "./datasets/cumulo-dc/"

TREE_DEPTH = 6
LR = 0.1
ITER = 2e4

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = CumuloDataset(path, ext="npz")
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=class_pixel_collate)

for d in dataloader:
    print(d["radiances"].size())
    print(d["properties"].size())
# train_stochastic(dataloader, )