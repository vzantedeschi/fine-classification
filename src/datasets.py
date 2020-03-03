import glob
import math
import numpy as np
import os

import netCDF4 as nc4

from torch.utils.data import Dataset

class Normalizer(object):

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, image):

        return (image - self.mean) / self.std

# ------------------------------------------------------------ CUMULO HELPERS

radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
coordinates = ['latitude', 'longitude']
properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties', 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity', 'surface_temperature']
rois = 'cloud_mask'
labels = 'cloud_layer_type'

def get_most_frequent_label(labelmask, dim=0):

    labels = np.argmax(labelmask, 0).astype(float)

    # set label of pixels with no occurences of clouds to NaN
    labels[np.sum(labelmask, 0) == 0] = np.NaN

    return labels

def read_nc(nc_file):
    """return masked arrays, with masks indicating the invalid values"""
    
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')

    f_radiances = np.vstack([file.variables[name][:] for name in radiances])
    f_properties = np.vstack([file.variables[name][:] for name in properties])
    f_rois = file.variables[rois][0]
    f_labels = file.variables[labels][0]

    return f_radiances, f_properties, f_rois, f_labels

def read_npz(npz_file):

    file = np.load(npz_file)

    return file['radiances'], file['properties'], file['cloud_mask'], file['labels']

class CumuloDataset(Dataset):

    def __init__(self, root_dir, normalizer=None, ext="npz"):
        
        self.root_dir = root_dir
        self.ext = ext

        self.file_paths = glob.glob(os.path.join(root_dir, "*." + ext))

        if len(self.file_paths) == 0:
            raise FileNotFoundError("no " + ext + " files in", self.root_dir)

        self.normalizer = normalizer

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):

        filename = self.file_paths[idx]
        
        if self.ext == "npz":
            radiances, properties, rois, labels = read_npz(filename)

        elif self.ext == "nc":
            radiances, properties, rois, labels = read_nc(filename)

        if self.normalizer is not None:
            radiances = self.normalizer(radiances)

        return radiances, properties, rois, labels

    def __str__(self):
        return 'CUMULO'