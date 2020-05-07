import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch

from src.datasets import toy_dataset
from src.optimization import train

DISTR = "swissroll"
N = 1000
TREE_DEPTH = 6
LR = 0.1
ITER = 2e4

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# generate toy dataset
X, Y = toy_dataset(N, DISTR)

# add offset
x = np.hstack((X, np.ones((N, 1))))

# train latent class tree and logistic regressor
model = train(x, Y, bst_depth=TREE_DEPTH, nb_iter=ITER, lr=LR)

# define colors
colors = [(1, 1, 1), (0.5, 0.5, 1)]
cm = LinearSegmentedColormap.from_list('twocolor', colors, N=100)

# create a mesh to plot in (points spread uniformly over the space)
H = .02  # step size in the mesh
x1_min, x1_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
x2_min, x2_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H)) # test points

# estimate learned class boundaries
test_x = np.c_[xx.ravel(), yy.ravel()]
test_x = np.hstack((test_x, np.ones((len(test_x), 1))))

t_x = torch.from_numpy(test_x).float()

y_pred = model.predict(t_x).numpy()
y_pred = y_pred.reshape(xx.shape)

plt.contourf(xx, yy, y_pred, cmap=cm, alpha=0.6)

# plot training points with true label
plt.scatter(X[Y == 0][:,0], X[Y == 0][:,1], cmap=cm, s=20, marker="o", edgecolors=colors[1], c=colors[0])
plt.scatter(X[Y == 1][:,0], X[Y == 1][:,1], cmap=cm, s=20, marker="^", c=colors[1])

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("{} dataset. lr={}; tree depth={}; iters={}".format(DISTR, LR, TREE_DEPTH, ITER))

plt.savefig("./results/latent-trees/" + DISTR + ".pdf", bbox_inches='tight', transparent=True)

plt.show()