import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import class_pixel_collate, compute_scaler, CumuloDataset, Scaler
from src.optimization import evaluate, LinearRegressor, train_stochastic
from src.utils import load_model, save_model

path = "./datasets/cumulo-dc/"

TREE_DEPTH = 2
LR = 5e-3
EPOCHS = 10

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# load CUMULO, all radiances and LWP property
dataset = CumuloDataset(path, ext="npz", prop_idx=[0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)
rad_preproc = compute_scaler(dataloader)
prop_preproc = Scaler(np.array([0]), np.array([6000]))

# reload data but rescaled and split in train, val, test
dataset = CumuloDataset(path, rad_preproc=rad_preproc, prop_preproc=prop_preproc, ext="npz", prop_idx=[0])

nb_tiles = len(dataset)
train_size, test_size = int(0.7*nb_tiles), int(0.2*nb_tiles)
val_size = nb_tiles - train_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# data loaders
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=class_pixel_collate)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=class_pixel_collate)

# 13 features, 1 property
model = LinearRegressor(TREE_DEPTH, 13, 1, 1)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# init loss
criterion = torch.nn.MSELoss(reduction="mean")

state = {
    'tile-size': 1,
    'batch-size': 1,
    'classifier': 'linear',
    'loss-function': 'MSE',
    'learning-rate': LR,
    'seed': SEED,
    'tree-depth': TREE_DEPTH,
    'dataset': path,
    'properties': 'LWP', 
}

best_val_loss = float("inf")

for e in range(EPOCHS):
    train_stochastic(trainloader, model, optimizer, criterion)

    val_loss = evaluate(valloader, model, criterion)
    print("Epoch %i: validation loss = %f\n" % (e, val_loss))

    if val_loss <= best_val_loss:
        best_val_loss = min(val_loss, best_val_loss)
        best_e = e
        save_model(model, optimizer, state, "./results/latent-trees/")

print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss))

best_model = load_model("./results/latent-trees/")
test_loss = evaluate(testloader, best_model, criterion)

print("test loss (epoch {}): {}".format(best_e, test_loss))
