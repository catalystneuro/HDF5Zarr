import argparse
import numpy as np
from hdf5zarr import HDF5Zarr
import json
import fsspec
import time

parser = argparse.ArgumentParser()
parser.add_argument('--selection', type=lambda s : eval('np.index_exp['+s+']'), default=':')
parser.add_argument('--dsetname', type=str, default='')
parser.add_argument('--url', type=str, default='')
args = parser.parse_args()
indexing = args.selection
dsetname = args.dsetname
url = args.url

metadata_file = 'metadata'
with open(metadata_file, 'r') as mfile:
    store = json.load(mfile)

start_time_file = time.time()
f = fsspec.open(url, 'rb').open()
hdf5_zarr = HDF5Zarr(filename = f, store=store, store_mode='r')
zgroup = hdf5_zarr.zgroup
end_time_file = time.time()

start_time_dset = time.time()
zgroup_data = zgroup[dsetname]
end_time_dset = time.time()
start_time = time.time()
arr= zgroup_data[indexing]
end_time = time.time()
print(f'time hdf5zarr_https {end_time-start_time}')
print(f'time hdf5zarr_https instantiate dataset {end_time_dset-start_time_dset}')
print(f'time hdf5zarr_https instantiate file {end_time_file-start_time_file}')
