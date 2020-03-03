import glob
import numpy as np
import os

from src.datasets import CumuloDataset
from src.utils import make_directory, TileExtractor

def load_labels(dir_name, image_name):

    label_name = "MYD021KM.{}{}.{}.*.npy".format(*image_name.split("."))
    label_name = glob.glob(os.path.join(dir_name, label_name))[0]

    return np.load(label_name)

load_path = "data/nc/"
save_path = "data/npz/"
label_path = "data/iresnet-labels/"

extractor = TileExtractor()
dataset = CumuloDataset(load_path, ext="nc")

make_directory(save_path)

for instance in dataset:

    name, *image = instance
    
    name = os.path.basename(name).replace(".nc", "")

    labels = load_labels(label_path, name)

    tiles, locations = extractor((*image, labels[None, ]))

    for i, (tile, loc) in enumerate(zip(tiles, locations)):

        save_name = os.path.join(save_path, "{}.{}".format(name, i))
        np.savez_compressed(save_name, radiances=tile[0].data, properties=tile[1].data, cloud_mask=tile[2].data, labels=tile[4], location=loc)
