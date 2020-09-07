<strong>Reading HDF5 files with Zarr</strong> building upon [Cloud-Performant NetCDF4/HDF5 Reading with the Zarr Library](https://medium.com/pangeo/cloud-performant-reading-of-netcdf4-hdf5-data-using-the-zarr-library-1a95c5c92314). This allows for efficiently reading HDF5 files stored remotely, and integration with Zarr-based computation tools.

[![codecov](https://codecov.io/gh/catalystneuro/HDF5Zarr/branch/master/graph/badge.svg)](https://codecov.io/gh/catalystneuro/HDF5Zarr)

## Installation:

Requires latest dev installation of h5py, with HDF5>=1.10.5.

### Install HDF5
Check available HDF5 version:
```bash
$ h5cc -showconfig
```

##### Conda:
``` bash
$ conda install "hdf5>=1.10.5"
```

##### Source installation:
Download and install [HDF5](https://www.hdfgroup.org/downloads/hdf5/)  
e.g.
```bash
$ cd hdf5*/bin
$ ./h5redeploy
```

### Install h5py
Follow h5py instructions for [custom installation](https://h5py.readthedocs.io/en/stable/build.html#custom-installation)  
For example:
##### Conda:
``` bash
$ HDF5_DIR=$CONDA_PREFIX pip install --no-binary=h5py git+https://github.com/h5py/h5py.git
```

##### Source installation:
```bash
$ HDF5_DIR=/path/to/hdf5 pip install --no-binary=h5py git+https://github.com/h5py/h5py.git
```

### Install HDF5Zarr

```bash
$ pip install git+https://github.com/catalystneuro/HDF5Zarr.git
```


## Usage:

## Reading local data
HDF5Zarr can be used to read a local HDF5 file where the datasets are actually read using the Zarr library.
Download example dataset from https://girder.dandiarchive.org/api/v1/item/5eda859399f25d97bd27985d/download:

```python

import requests
import os.path as op
file_name = 'sub-699733573_ses-715093703.nwb'

if not op.exists(file_name):
    response = requests.get("https://girder.dandiarchive.org/api/v1/item/5eda859399f25d97bd27985d/download")
    with open(file_name, mode='wb') as localfile:
        localfile.write(response.content)

```

```python

import zarr
from hdf5zarr import HDF5Zarr

file_name = 'sub-699733573_ses-715093703.nwb'
hdf5_zarr = HDF5Zarr(filename = file_name, store_mode='w', max_chunksize=2*2**20)
zgroup = hdf5_zarr.consolidate_metadata(metadata_key = '.zmetadata')
```
Without indicating a specific zarr store, zarr uses the default `zarr.MemoryStore`.
Alternatively, pass a zarr store such as:
```python
store = zarr.DirectoryStore('storezarr')
hdf5_zarr = HDF5Zarr(file_name, store = store, store_mode = 'w')
```

Examine structure of file using Zarr tools:
```python
# print dataset names
zgroup.tree()
# read
arr = zgroup['units/spike_times']
val = arr[0:1000]
```

Once you have a zgroup object, this object can be read by PyNWB using
```python
from hdf5zarr import NWBZARRHDF5IO
io = NWBZARRHDF5IO(mode='r+', file=zgroup)
```

Export metadata from zarr store to a single json file
```python
import json
metadata_file = 'metadata'
with open(metadata_file, 'w') as mfile:
    json.dump(zgroup.store.meta_store, mfile)
```


## Open NWB file on remote S3 store.
Requires a local metadata_file, constructed in previous steps.
```python
import s3fs
from hdf5zarr import NWBZARRHDF5IO


# import metadata from a json file
with open(metadata_file, 'r') as mfile:
    store = json.load(mfile)

fs = s3fs.S3FileSystem(anon=True)

f = fs.open('dandiarchive/girder-assetstore/4f/5a/4f5a24f7608041e495c85329dba318b7', 'rb')

hdf5_zarr = HDF5Zarr(f, store = store, store_mode = 'r')
zgroup = hdf5_zarr.zgroup
io = NWBZARRHDF5IO(mode='r', file=zgroup, load_namespaces=True)
```

Here is the entire workflow for opening a file remotely:
```python
import zarr
import s3fs
from hdf5zarr import HDF5Zarr, NWBZARRHDF5IO

file_name = 'sub-699733573_ses-715093703.nwb'
store = zarr.DirectoryStore('storezarr')
hdf5_zarr = HDF5Zarr(filename = file_name, store=store, store_mode='w', max_chunksize=2*2**20)
zgroup = hdf5_zarr.consolidate_metadata(metadata_key = '.zmetadata')


fs = s3fs.S3FileSystem(anon=True)

f = fs.open('dandiarchive/girder-assetstore/4f/5a/4f5a24f7608041e495c85329dba318b7', 'rb')
hdf5_zarr = HDF5Zarr(f, store = store, store_mode = 'r')
zgroup = hdf5_zarr.zgroup
io = NWBZARRHDF5IO(mode='r', file=zgroup, load_namespaces=True)
nwb = io.read()
```
