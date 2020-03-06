import math
import numpy as np
import os

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_as_npy(data, flag, dirname, deep=True):

    make_directory(dirname)

    if deep:
        for key, item in data.items():
            np.save(os.path.join(dirname, "{}-{}.npy".format(flag, key)), item)
    else:
        np.save(os.path.join(dirname, "{}.npy".format(flag)), data)


class TileExtractor(object):

    def __init__(self, t_width=256, t_height=256):

        self.t_width = t_width
        self.t_height = t_height

    def __call__(self, image):
        """ 
        Parameters
        ----------
        image : tuple of numpy arrays or masked arrays of shape (., width, height, ...)

        """

        img_width = image[0].shape[1]
        img_height = image[0].shape[2]

        nb_tiles_row = math.ceil(img_width / self.t_width)
        nb_tiles_col = math.ceil(img_height / self.t_height)

        tile_locations = []

        for i in range(nb_tiles_row):
            for j in range(nb_tiles_col):
                
                x2, y2 = min(img_width, (i+1) * self.t_width), min(img_height, (j+1) * self.t_height)
                x1, y1 = x2 - self.t_width, y2 - self.t_height

                tile_locations.append((x1, x2, y1, y2))

        tiles = []
        for i, (x1, x2, y1, y2) in enumerate(tile_locations):
            tiles.append([array[:, x1: x2, y1: y2] for array in image])

        return tiles, tile_locations