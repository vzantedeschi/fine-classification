import matplotlib.pyplot as plt
import numpy as np

import torch

from src.datasets import toy_dataset
from src.optimization import train

N = 1000
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)

# generate XOR dataset
X, Y = toy_dataset(N)

# add offset
x = np.hstack((X, np.ones((N, 1))))

# train latent class tree and logistic regressor
model = train(x, Y)

# predict on training data
t_x = torch.from_numpy(x).float()
y_pred = model.predict(t_x).numpy()

# create a mesh to plot in
H = .02  # step size in the mesh
x1_min, x1_max = X[:,0].min()-0.2, X[:,0].max()+0.2
x2_min, x2_max = X[:,1].min()-0.2, X[:,1].max()+0.2
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H))

plt.scatter(X[:,0], X[:,1], cmap=plt.cm.coolwarm, s=20, c=y_pred)

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("xor")

plt.show()