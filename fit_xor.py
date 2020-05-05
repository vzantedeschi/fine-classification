import matplotlib.pyplot as plt
import numpy as np

from src.datasets import toy_dataset

X, Y = toy_dataset()

# create a mesh to plot in
H = .02  # step size in the mesh
x1_min, x1_max = X[:,0].min()-0.2, X[:,0].max()+0.2
x2_min, x2_max = X[:,1].min()-0.2, X[:,1].max()+0.2
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, H), np.arange(x2_min, x2_max, H))

plt.scatter(X[:,0], X[:,1], cmap=plt.cm.coolwarm, s=20, c=Y)

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.title("xor")

plt.show()