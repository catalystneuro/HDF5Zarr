<strong>Reading HDF5 files with Zarr</strong>

## Installation

```bash
$ pip install git+https://github.com/catalystneuro/allen-institute-neuropixel-utils
```

## Usage:

```python

from hdf5zarr import HDF5Zarr
from hdf5zarr import rewrite_vlen_to_fixed
import zarr
import fsspec

file_name = 'ecephys.nwb'

# Optional, if hdf5 file contains variable-length string datasets, rewrite them as fixed-length
rewrite_vlen_to_fixed(file_name)

# Local read
hdf5_zarr = HDF5Zarr(file_name)
# Without indicating a specific zarr store, zarr uses the default zarr.MemoryStore
# alternatively pass a zarr store such as:
# store = zarr.DirectoryStore('storezarr')
# hdf5_zarr = HDF5Zarr(file_name, store = store, store_mode = 'w')
zgroup = hdf5_zarr.consolidate_metadata(metadata_key = '.zmetadata')
# print dataset names
zgroup.tree()
# read
arr = zgroup['units/spike_times']
val = arr[0:1000]

# export metadata from zarr store to a single json file
import json
metadata_file = 'metadata'
with open(metadata_file, 'w') as f:
    json.dump(zgroup.store.meta_store, f)

# Remote read
import s3fs
# Set up S3 access
fs = s3fs.S3FileSystem()

# import metadata from a json file
with open(metadata_file, 'r') as f:
    metadata_dict = json.load(f)

store = metadata_dict
with fs.open('bucketname/' + file_name, 'rb') as f:
    hdf5_zarr = HDF5Zarr(f, store = store, store_mode = 'r')
    zgroup = hdf5_zarr.zgroup
    # print dataset names
    zgroup.tree()
    arr = zgroup['units/spike_times']
    val = arr[0:1000]

```
