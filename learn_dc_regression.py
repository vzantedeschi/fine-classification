import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import class_pixel_collate, compute_scaler, CumuloDataset, Scaler, property_ranges, properties
from src.monitors import MonitorTree
from src.optimization import NonLinearRegression, evaluate, train_stochastic
from src.property_analysis import distributions_from_labels, compute_bins
from src.utils import load_model, save_as_npz, save_model

dataset_path = "./datasets/cumulo-dc/"

# LWP = 0, COT = 1, CTP = 4, ST = 8
TRAIN_PROP = [0, 4]
TEST_PROP = [8]

bins = compute_bins([[-1, 1, 51], [-1, 1, 51], [-1, 1, 51]])

LR = 1e-3
EPOCHS = 100

save_dir = "./results/dc-subtypes/rad-LWP-CTP/nonlin-regression/"

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# load CUMULO, all radiances and selected properties
dataset = CumuloDataset(dataset_path, ext="npz", prop_idx=TRAIN_PROP, test_prop_idx=TEST_PROP)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=class_pixel_collate)

rad_preproc = compute_scaler(dataloader)
prop_preproc = Scaler(property_ranges[0, TRAIN_PROP], property_ranges[1, TRAIN_PROP])
test_prop_preproc = Scaler(property_ranges[0, TEST_PROP], property_ranges[1, TEST_PROP])

# reload data but rescaled and split in train, val, test
dataset = CumuloDataset(dataset_path, ext="npz", rad_preproc=rad_preproc, prop_preproc=prop_preproc, test_prop_preproc=test_prop_preproc, prop_idx=TRAIN_PROP, test_prop_idx=TEST_PROP)

nb_tiles = len(dataset)
train_size, test_size = int(0.7*nb_tiles), int(0.2*nb_tiles)
val_size = nb_tiles - train_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# data loaders
trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=class_pixel_collate)
valloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=class_pixel_collate)
testloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=class_pixel_collate)

# 13 features, train properties => test properties
model = NonLinearRegression(13 + len(TRAIN_PROP), len(TEST_PROP))

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# init loss
criterion = torch.nn.MSELoss(reduction="mean")

# init train-eval monitoring 
monitor = MonitorTree(False, save_dir)

state = {
    'tile-size': 1,
    'batch-size': 2,
    'loss-function': 'MSE',
    'learning-rate': LR,
    'seed': SEED,
    'dataset': dataset_path,
    'features': 'radiances-LWP-CTP', 
    'target': 'ST',
}

best_val_loss = float("inf")
best_e = -1
for e in range(EPOCHS):
    train_stochastic(trainloader, model, optimizer, criterion, pruning=False, epoch=e, monitor=monitor)

    val_loss = evaluate(valloader, model, criterion, epoch=e, monitor=monitor)
    print("Epoch %i: validation loss = %f\n" % (e, val_loss))

    if val_loss <= best_val_loss:
        best_val_loss = min(val_loss, best_val_loss)
        best_e = e
        save_model(model, optimizer, state, save_dir)

monitor.close()
print("best validation loss (epoch {}): {}\n".format(best_e, best_val_loss))

best_model = load_model(save_dir)
test_loss = evaluate(testloader, best_model, criterion)

print("test loss (epoch {}): {}".format(best_e, test_loss)) 