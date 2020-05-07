import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import CumuloDataset, class_pixel_collate
from src.optimization import LinearRegressor, train_stochastic

path = "./datasets/cumulo-dc/"

TREE_DEPTH = 2
LR = 0.1
EPOCHS = 10

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# load CUMULO, all radiances and LWP property
dataset = CumuloDataset(path, ext="npz", prop_idx=[0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)

# 13 features, 1 property
model = LinearRegressor(TREE_DEPTH, 13, 1)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# init loss
criterion = torch.nn.MSELoss(reduction="mean")

for e in range(EPOCHS):
    train_stochastic(dataloader, model, optimizer, criterion)