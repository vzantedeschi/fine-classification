import numpy as np
import os
from shutil import copyfile

from src.datasets import CumuloDataset
from src.utils import make_directory

load_path = "data/npz/"
save_path = "datasets/cumulo-dc/"

THR = 1000 # select tiles that contain at least <threshold> pixels of class Deep Convection
LABEL = 7 # index corresponding to coarse class Deep Convection

dataset = CumuloDataset(load_path, ext="npz")

make_directory(save_path)

for instance in dataset:

    name, *_, mask, labels = instance

    dc_pixels = np.sum(np.logical_and(labels == LABEL, mask))

    if dc_pixels > THR:

        print(name, dc_pixels)
        copyfile(name, os.path.join(save_path, os.path.basename(name)))