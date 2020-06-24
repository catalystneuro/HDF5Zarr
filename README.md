<strong>Reading HDF5 files with Zarr</strong>

## Installation

Requires latest dev installation of h5py


```bash
$ pip install git+https://github.com/catalystneuro/allen-institute-neuropixel-utils
```


## Usage:

## Reading local data
HDF5Zarr can be used to read a local HDF5 file where the datasets are actually read using the Zarr library.

```python
import zarr
from hdf5zarr import HDF5Zarr

file_name = '/Users/bendichter/dev/allen-institute-neuropixel-utils/sub-699733573_ses-715093703.nwb'
store = zarr.DirectoryStore('storezarr')
hdf5_zarr = HDF5Zarr(filename = file_name, store=store, store_mode='w', max_chunksize=2*2**20)
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

Once you have a zgroup object, this object can be read by PyNWB using the 
```python
from hdf5zarr import NWBZARRHDF5IO
io = NWBZARRHDF5IO(mode='r+', file=zgroup)     
```

        
Open NWB file on remote S3 store. requires a loyal metadata_file, constructed in previous steps.
```python
import s3fs
from hdf5zarr import NWBZARRHDF5IO


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

file_name = '/Users/bendichter/dev/allen-institute-neuropixel-utils/sub-699733573_ses-715093703.nwb'
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
