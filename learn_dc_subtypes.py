import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import class_pixel_collate, compute_scaler, CumuloDataset, Scaler
from src.optimization import evaluate, LinearRegressor, train_stochastic
from src.property_analysis import distributions_from_labels, compute_bins
from src.utils import load_model, save_as_npy, save_model

path = "./datasets/cumulo-dc/"
save_dir = "./results/latent-trees/"

TREE_DEPTH = 2
LR = 1e-3
EPOCHS = 200
nb_classes = 2**TREE_DEPTH

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# load CUMULO, all radiances and LWP property
dataset = CumuloDataset(path, ext="npz", prop_idx=[0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)
rad_preproc = compute_scaler(dataloader)
prop_preproc = Scaler(np.array([0]), np.array([5000]))

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
        save_model(model, optimizer, state, save_dir)

print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss))

best_model = load_model(save_dir)
train_loss, train_labels, train_properties = evaluate(trainloader, best_model, criterion, classify=True)
val_loss, val_labels, val_properties = evaluate(valloader, best_model, criterion, classify=True)
test_loss, test_labels, test_properties = evaluate(testloader, best_model, criterion, classify=True)

print("test loss (epoch {}): {}".format(best_e, test_loss))
print(np.histogram(train_labels, range(nb_classes+1)), np.histogram(val_labels, range(nb_classes+1)), np.histogram(test_labels, range(nb_classes+1)))

# define LWP bins
bins = compute_bins([[0, 1, 51]])

distr_joint, distr_c, distr_s = {}, {}, {}
distr_joint["train"], distr_c["train"], distr_s["train"] = distributions_from_labels(train_properties.T, train_labels, bins, nb_classes=nb_classes)
distr_joint["val"], distr_c["val"], distr_s["val"] = distributions_from_labels(val_properties.T, val_labels, bins, nb_classes=nb_classes)
distr_joint["test"], distr_c["test"], distr_s["test"] = distributions_from_labels(test_properties.T, test_labels, bins, nb_classes=nb_classes)

save_as_npy(distr_joint, "P(s,c)", save_dir)
save_as_npy(distr_s, "P(s)", save_dir)
save_as_npy(distr_c, "P(c)", save_dir)
save_as_npy(bins, "bins", save_dir, deep=False)