import argparse
import numpy as np
from hdf5zarr import HDF5Zarr
import h5py
import time


parser = argparse.ArgumentParser()
parser.add_argument('--selection', type=lambda s : eval('np.index_exp['+s+']'), default=':')
parser.add_argument('--dsetname', type=str, default='')
parser.add_argument('--url', type=str, default='')
args = parser.parse_args()
indexing = args.selection
dsetname = args.dsetname
url = args.url

start_time_file = time.time()
hf = h5py.File(url, 'r', driver='ros3')
end_time_file = time.time()
start_time_dset = time.time()
hgroup = hf[dsetname]
end_time_dset = time.time()
start_time = time.time()
arr = hgroup[indexing]
end_time = time.time()
print(f'time h5py_ros3 {end_time-start_time}')
print(f'time h5py_ros3 instantiate dataset {end_time_dset-start_time_dset}')
print(f'time h5py_ros3 instantiate file {end_time_file-start_time_file}')
