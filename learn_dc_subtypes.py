import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import class_pixel_collate, compute_scaler, CumuloDataset, Scaler
from src.optimization import LinearRegressor, train_stochastic

path = "./datasets/cumulo-dc/"

TREE_DEPTH = 2
LR = 5e-3
EPOCHS = 100

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# load CUMULO, all radiances and LWP property
dataset = CumuloDataset(path, ext="npz", prop_idx=[0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)
rad_preproc = compute_scaler(dataloader)
prop_preproc = Scaler(np.array([0]), np.array([6000]))

# reload data but rescaled
dataset = CumuloDataset(path, rad_preproc=rad_preproc, prop_preproc=prop_preproc, ext="npz", prop_idx=[0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)

# 13 features, 1 property
model = LinearRegressor(TREE_DEPTH, 13, 1, 1)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# init loss
criterion = torch.nn.MSELoss(reduction="mean")

for e in range(EPOCHS):
    train_stochastic(dataloader, model, optimizer, criterion)