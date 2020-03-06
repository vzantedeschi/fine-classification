import numpy as np
from sklearn.model_selection import train_test_split

from src.classification import train_booster, predict_booster, save_model
from src.datasets import CumuloDataset
from src.property_analysis import distributions_from_labels, compute_bins
from src.utils import save_as_npy

save_dir = "results/lightgbm/"

# training parameters
RND = 42 # random state
TEST = 0.2 # test set's size
VAL = 0.1 # validation set's size

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'verbose': 1,
    'num_classes': 2,
    "num_iterations": 1000,
}

data_dir = 'datasets/cumulo-dc/'
dataset = CumuloDataset(root_dir=data_dir, ext="npz")

# flat dataset for pixel-wise classification
xs = [] # features
lwps = [] # continuos output Liquid Water Path

for name, radiances, properties, mask, labels in dataset:

    dc_mask = labels[0] == 7
    xs.append(radiances[:,dc_mask].T)
    lwps.append(properties[0][dc_mask].T)

xs = np.vstack(xs)
lwps = np.hstack(lwps)
# binarize problem
ys = lwps > 2000

train_xs, test_xs, train_ys, test_ys, train_lwps, test_lwps = train_test_split(xs, ys, lwps, test_size=TEST, random_state=RND)
train_xs, val_xs, train_ys, val_ys, train_lwps, val_lwps = train_test_split(train_xs, train_ys, train_lwps, test_size=VAL/(1-TEST), random_state=RND)

model = train_booster(train_xs, train_ys, val_xs, val_ys, params)

train_y_pred = predict_booster(model, train_xs)
val_y_pred = predict_booster(model, val_xs)
test_y_pred = predict_booster(model, test_xs)

# define LWP bins
bins = compute_bins([[0, 4000, 51]])

distr_joint, distr_c, distr_s = {}, {}, {}
distr_joint["train"], distr_c["train"], distr_s["train"] = distributions_from_labels([train_lwps], train_y_pred, bins, nb_classes=2)
distr_joint["val"], distr_c["val"], distr_s["val"] = distributions_from_labels([val_lwps], val_y_pred, bins, nb_classes=2)
distr_joint["test"], distr_c["test"], distr_s["test"] = distributions_from_labels([test_lwps], test_y_pred, bins, nb_classes=2)

save_as_npy(distr_joint, "P(s,c)", save_dir)
save_as_npy(distr_s, "P(s)", save_dir)
save_as_npy(distr_c, "P(c)", save_dir)
save_as_npy(bins, "bins", save_dir, deep=False)

save_model(model, save_dir)