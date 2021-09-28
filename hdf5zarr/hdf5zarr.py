import h5py
import zarr
from zarr.storage import array_meta_key, group_meta_key, attrs_key
from zarr.storage import ConsolidatedMetadataStore
import numpy as np
import numcodecs
import fsspec
from fsspec.implementations import local
from fsspec.spec import AbstractBufferedFile
from fsspec.asyn import AsyncFileSystem
from typing import Union
import os
import io
from pathlib import Path
from collections.abc import MutableMapping
from pathlib import PurePosixPath
from zarr.util import json_dumps, json_loads, normalize_storage_path
from xdrlib import Unpacker
import struct
from sys import stdout
from warnings import warn
from functools import partial
import copy
import types
import tempfile
import re
import time
SYMLINK = '.link'

from dask.distributed import Client, as_completed, get_task_stream, LocalCluster, Worker
import logging
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "False"


class UnpackerVlenString(Unpacker):

    def reset(self, data):
        self.__buf = data
        self.__pos = 0

    def unpack_fopaque(self, n, size=8):
        if n < 0:
            raise ValueError('fstring size must be nonnegative')
        i = self.__pos
        j = i + (n+(size-1))//size*size
        if j > len(self.__buf):
            return 0
        self.__pos = j
        return self.__buf[i:i+n]

    def unpack_opaque(self):
        nid = self.unpack_uint64()
        n = self.unpack_uint64()
        b = self.unpack_fopaque(n, size=8)
        if nid == 0 or b == 0:
            # end of gcol
            return None
        return nid, b

    def unpack_uint64(self, n=8):
        i = self.__pos
        self.__pos = j = i+n
        data = self.__buf[i:j]
        if len(data) < n:
            # end of gcol
            return 0
        return struct.unpack('<Q', data)[0]

    def unpack_vlenstring(self, unpack_item):
        vlen_list = []
        id_list = []
        while 1:
            item = unpack_item()
            if item is None:
                return id_list, vlen_list
            id_list.append(item[0])
            vlen_list.append(item[1])
        return id_list, vlen_list


class VLenHDF5String(numcodecs.abc.Codec):

    codec_id = 'VLenHDF5String'

    def __init__(self, size=8):
        self.size = size
        # TO DO #
        self.dt_vlen = np.dtype([('size', 'uint32'), ('address', f'uint{self.size*8}'), ('id', 'uint32')])

    def decode(self, buf, out=None):

        data_array = np.frombuffer(buf[0], dtype=self.dt_vlen)
        data_offsets = np.unique(data_array['address'])
        data_offsets.sort()
        sort_args = np.argsort(data_array[['address', 'id']])
        address_sorted = data_array['address'][sort_args]
        ids_sorted = data_array['id'][sort_args]
        p = np.where(np.diff(address_sorted, prepend=-1, append=np.inf))[0]

        vlen_array = np.empty(shape=len(data_array), dtype=object)
        for i in range(len(data_offsets)):
            unpacker = UnpackerVlenString(buf[i+1])
            id_list, vlen_list = unpacker.unpack_vlenstring(unpacker.unpack_opaque)
            ids_sorted_i = ids_sorted[p[i]:p[i+1]]
            sorter = np.argsort(id_list)
            vlen_str_index = np.searchsorted(id_list, ids_sorted_i, sorter=sorter)
            vlen_array[p[i]:p[i+1]] = np.chararray.decode(np.array(vlen_list)[sorter[vlen_str_index]], encoding='utf-8')

        vlen_array = vlen_array[np.argsort(sort_args)]

        return vlen_array

    def encode(self, buf):
        raise RuntimeError('VLenHDF5String: Cannot encode')

    def get_config(self):
        return {'id': self.codec_id, 'size': self.size}

    @classmethod
    def from_config(cls, config):
        size = config['size']
        return cls(size)


numcodecs.register_codec(VLenHDF5String)


def open_as_zarr(filename, **kwargs):
    """ open zarr group via HDF5Zarr
    Args:
        filename:                str, h5py dataset, group or file, or File-like object to be read by zarr
        collectattrs:            bool, whether to collect attributes or not, default True
        collectrefs:             bool or list, passing a dataset with object reference dtype without setting collectrefs
                                 will dereference dataset to null when read with zarr.
                                 set to a list of absolute paths in hdf5 file that the dataset may refers to,
                                 set to True to collect all references for all objects in the hdf5 file.
        **kwargs:                keyword arguments passed to HDF5Zarr
    """

    hdf5zarr = HDF5Zarr(filename, **kwargs)
    zgroup = hdf5zarr.zgroup
    return zgroup


def export_to_zarr(filename, zarrstore: Union[MutableMapping, str, Path], **kwargs):
    """ read hdf5 data via HDF5Zarr, and copy to zarrstore
    Args:
        filename:                str, h5py dataset, group or file, or File-like object to be read by zarr
        zarrstore:               zarr store to export data to,
                                 collections.abc.MutableMapping or str,
                                 if string path is passed, zarr.DirectoryStore
        **kwargs:                keyword arguments passed to HDF5Zarr
    """

    hdf5zarr = HDF5Zarr(filename, **kwargs)
    zgroup = hdf5zarr.zgroup
    zout = zarr.open_group(zarrstore, mode='a')
    zarr.copy(zgroup, zout, name='/', log=stdout)


class HDF5Zarr(object):
    """ class to create zarr structure for reading hdf5 files """

    def __init__(self, filename: str, hdf5obj: str = None,
                 collectattrs: bool = True, uri: str = None, hdf5group: str = None,
                 store: Union[MutableMapping, str, Path] = None, store_path: str = None,
                 store_mode: str = 'a', LRU: bool = False, LRU_max_size: int = 2**30,
                 max_chunksize=2*2**20, driver: str = None, blocksize: int = 2**15, collectrefs: Union[bool, str, list] = None,
                 withdask: bool = True, dask_kwds: dict = {}, follow_symlinks: bool = True):

        """
        Args:
            filename:                    str, File-like, h5py file, group or dataset object to be read by zarr
            hdf5group:                   str, hdf5 group in hdf5 file to be read by zarr
                                         along with its children. default is the root group.
            hdf5obj:                     same as hdf5group for accepting either dataset or group,
                                         overrides hdf5group
            collectattrs:                bool, whether to collect attributes or not, default True
            collectrefs:                 affects only if filename is not an h5py.File,
                                         or if hdf5obj points to a dataset or group
                                         bool or list, whether to collect object references
                                         set to True to collect all references
                                         set to list of paths to only collect references for objects in the list
            uri:                         set uri in zarr store,
                                         overrides determining uri from filename
            store:                       collections.abc.MutableMapping or str, zarr store.
                                         if string path is passed, zarr.DirectoryStore
                                         is created at the given path, if None, zarr.MemoryStore is used
            store_mode:                  store data access mode, default 'a'
                                         'r'          readonly, compatible zarr hierarchy should
                                                      already exist in the passed store
                                         'r+'         read and write, return error if file does not exist,
                                                      for updating zarr hierarchy
                                         'w'          create store, remove data if it exists
                                         'w-' or 'x'  create store, fail if exists
                                         'a'          read and write, create if it does not exist, default 'a'
            store_path:                  string, path in zarr store
            LRU:                         bool, if store is not already zarr.LRUStoreCache, add
                                         a zarr.LRUStoreCache store layer on top of currently used store
            LRU_max_size:                int, maximum zarr.LRUStoreCache cache size, only used
                                         if store is zarr.LRUStoreCache, or LRU argument is True
            max_chunksize:               maximum chunk size to use when creating zarr hierarchy, this is useful if
                                         only a small slice of data needs to be read
            driver:                      driver name to pass to h5py
            blocksize:                   affects only if filename is opened with fsspec,
                                         used for filename cache blocksize only when building metadata,
                                         This is not used for chunkstore when reading data,
                                         default 16kb
            withdask                     default True
            dask_kwds                    default {'silence_logs': logging.WARNING,
                                                  'dashboard_address': None,
                                                  'processes': True,
                                                  'n_workers': 1,
                                                  'threads_per_worker': 1,
                                                  'worker_class': Worker,
                                                  }
            follow_symlinks              only applies when withdask is True, default True
        """

        if withdask:
            dask_opts = {'silence_logs': logging.WARNING,
                         'dashboard_address': None,
                         'processes': True, # turns on Nanny in LocalCluster
                         'n_workers': 1,
                         'threads_per_worker': 1,
                         'worker_class': Worker, # turn off Nanny when processes == True
                         }
            if not isinstance(dask_kwds, dict):
                raise TypeError(f"Expected dict for dask_kwds, recieved {type(dask_kwds)}")
            else:
                dask_opts.update(dask_kwds)
            cluster = LocalCluster(**dask_opts)  # start local workers as processes
            cluster.adapt(minimum = 1, minimum_cores = 1)
            self.client = Client(cluster)

        # Verify arguments
        if not isinstance(LRU, bool):
            raise TypeError(f"Expected bool for LRU, recieved {type(LRU)}")
        self.LRU = LRU
        if not isinstance(LRU_max_size, int):
            raise TypeError(f"Expected int for LRU_max_size, recieved {type(LRU_max_size)}")
        self.LRU_max_size = LRU_max_size
        if max_chunksize is not None and not (isinstance(max_chunksize, int) and max_chunksize > 0):
            raise TypeError(f"Expected positive int or None for max_chunksize,\
                              recieved {max_chunksize}, type: {type(max_chunksize)}")
        self.max_chunksize = max_chunksize
        if not isinstance(collectattrs, bool):
            raise TypeError(f"Expected bool for collectattrs, recieved {type(collectattrs)}")
        self.collectattrs = collectattrs
        if collectrefs is not None and (not isinstance(collectrefs, (bool, str, list)) or
                isinstance(collectrefs, list) and any(not isinstance(i, str) for i in collectrefs)):
            raise TypeError(f"Expected bool, str or list of strings for collectrefs, recieved {type(collectrefs)}")
        self.collectrefs = collectrefs
        if uri is not None and not isinstance(uri, str):
            raise TypeError(f"Expected str for uri, recieved {type(uri)}")
        self.uri = uri
        if not isinstance(blocksize, int) or blocksize <= 0:
            raise TypeError(f"Expected positive int for blocksize, recieved {type(blocksize)}")
        self.blocksize = blocksize
        self.follow_symlinks = follow_symlinks

        # store and store_mode are passed through to zarr
        self.store_mode = store_mode
        if store is not None and LRU is True and not isinstance(store, zarr.LRUStoreCache):
            self.store = zarr.LRUStoreCache(store, max_size=self.LRU_max_size)
        else:
            self.store = store

        # create dictionary mapping hdf5 filter numbers to compatible zarr codec
        HDF5Zarr._hdf5_regfilters_subset = {}
        self._fill_regfilters()

        # dictionary to hold addresses of hdf5 objects in file
        self._address_dict = {}

        # create zarr format hierarchy for datasets and attributes compatible with hdf5 file,
        # dataset contents are not copied, unless it contains variable-length strings

        if hdf5group is not None and (not isinstance(hdf5group, str) or len(hdf5group) == 0):
            raise TypeError(f"Expected non-empty str for hdf5group, recieved: {hdf5group}, {type(hdf5group)}")
        if hdf5obj is not None:
            if not isinstance(hdf5obj, str) or len(hdf5obj) == 0:
                raise TypeError(f"Expected non-empty str for hdf5obj, recieved: {hdf5obj}, {type(hdf5obj)}")
            hdf5group = hdf5obj

        self._filename = filename # reference to passed filename
        h5group_filename = False # indicates if filename passed is an h5py group object
        h5obj_filename = False # indicates if filename passed is an h5py object
        if isinstance(filename, (h5py.File, h5py.Group, h5py.Dataset)):
            if not isinstance(filename, h5py.File) and not (isinstance(filename, h5py.Group) and filename.name == '/'):
                # check keyword arguments
                if hdf5group and '/'+normalize_storage_path(hdf5group) != filename.name:
                    raise Exception(f'hdf5obj or hdf5group is specified, but it is '+
                                    f'different from {filename} with name {filename.name}, ambiguous arguments')
                hdf5group = filename.name
                h5group_filename = True

            self.group = filename
            self.file = filename.file
            filename = filename.file.filename
            if withdask:
                fs_h5obj, _, _ = fsspec.get_fs_token_paths(filename)
                if isinstance(fs_h5obj, local.LocalFileSystem):
                    try:
                        fs_h5obj.open(filename, mode='rb')
                    except Exception as ex:
                        print(f"can not access hdf5 object filename {filename}")
                        raise ex
            h5obj_filename = True
            if driver is not None:
                warn(f"driver {driver} has no effect. filename is an h5py.File")

        if hdf5group is not None and store_path is None:
            self.store_path = hdf5group  # store_path is passed to zarr
        else:
            self.store_path = store_path

        if withdask:
            store = self.store
            self.store = None
            self.zgroup = zarr.open(self.store, mode=self.store_mode)
            self.store = self.zgroup.store
        else:
            self.zgroup = zarr.open(self.store, mode=self.store_mode)
            if self.store is None:
                self.store = self.zgroup.store

            self.zgroup = zarr.open(self.store, mode=self.store_mode)

        # FileChunkStore requires uri
        if self.uri is None:
            if isinstance(filename, str):
                self.uri = filename
            else:
                try:
                    self.uri = getattr(filename, 'path', None)
                    if self.uri is None:
                        self.uri = filename.name
                    if self.uri in (None, b'') or len(str(self.uri)) == 0:
                        raise Exception
                except Exception:
                    warn('unable to determine uri. uri argument is not passed')
                    self.uri = str(filename)

        # Access hdf5 file and create zarr hierarchy
        self.hdf5group = hdf5group
        self.filename = filename
        if not withdask and isinstance(self.filename, str):
            # checks filename file system with fsspec
            try:
                fs, _, _ = fsspec.get_fs_token_paths(self.filename)
                if not isinstance(fs, local.LocalFileSystem):
                    tempfilename = re.sub(r'[^\w]', '', self.filename)
                    temp_file = tempfile.NamedTemporaryFile(suffix="", prefix='temp_'+tempfilename)
                    temp_file_name = temp_file.name
                    temp_file.close()
                    cache_type = 'mmap'  # TO DO
                    self.filename = fs.open(self.filename, mode='rb', cache_type=cache_type,
                                            block_size=self.blocksize,
                                            cache_options={'location':temp_file_name})
            except:
                pass

        if withdask:
            # checks filename file system with fsspec
            fs = False
            try:
                fs, _, _ = fsspec.get_fs_token_paths(self.filename)
            except:
                pass

            if isinstance(self.filename, local.LocalFileOpener):
                self.filename = self.filename.path # avoid unable to pickle in run_create_zarr_hierarchy
            if fs != False:
                if not isinstance(fs, local.LocalFileSystem):
                    try:
                        tempfilename = re.sub(r'[^\w]', '', self.filename)
                        temp_file = tempfile.NamedTemporaryFile(suffix="", prefix='temp_'+tempfilename)
                        temp_file_name = temp_file.name
                        temp_file.close()
                        cache_type = 'mmap'
                        self.filename = fs.open(self.filename, mode='rb', cache_type=cache_type,
                                                block_size=self.blocksize,
                                                cache_options={'location':temp_file_name})
                    except Exception as ex:
                        print(f"can not access filename {self.filename}")
                        raise ex
                elif not h5obj_filename:
                    try:
                        fs.open(self.filename, mode='rb')
                    except Exception as ex:
                        print(f"can not access filename {self.filename}")
                        raise ex

        if self.store_mode != 'r':
            if not h5obj_filename:
                # set self.file to h5py.File object
                if driver is None and not isinstance(self.filename, str):
                    #  checking for valid cache object in self.filename
                    _cached = False
                    try:
                        _cache = self.filename.cache
                        if isinstance(_cache, fsspec.caching.BaseCache):
                            _cached = True
                    except:
                        pass

                    self.file = h5py.File(self.filename, mode='r')
                    if _cached:
                        self.filename.cache = _cache
                else:
                    self.file = h5py.File(self.filename, mode='r', driver=driver)

            if not h5obj_filename:
                self.group = self.file[self.hdf5group] if self.hdf5group is not None else self.file

            self.hdf5group = self.hdf5group if self.hdf5group is not None else '/'
            if withdask:
                self.run_create_zarr_hierarchy(self.hdf5group)
            else:
                self.create_zarr_hierarchy(self.group, self.zgroup)
            if not h5obj_filename:
                self.file.close()

            if withdask:
                if store is not None:
                    try:
                        store.update(self.store)
                        self.store = store
                    except:
                        warn(f"failed updating passed store {store}, returning zarr object with default store")
                        pass

        if isinstance(self.filename, str):
            self.chunkstore_file = fsspec.open(self.filename, mode='rb')
            self.chunk_store = FileChunkStore(self.store, chunk_source=self.chunkstore_file.open())
        else:
            self.chunk_store = FileChunkStore(self.store, chunk_source=self.filename)
        if LRU is True and not isinstance(self.chunk_store, zarr.LRUStoreCache):
            self.chunk_store = zarr.LRUStoreCache(self.chunk_store, max_size=self.LRU_max_size)

        # open zarr group
        store_mode_cons = 'r' if self.store_mode == 'r' else 'r+'
        self.zgroup = zarr.open(self.store, mode=store_mode_cons, path=self.store_path, chunk_store=self.chunk_store)
        if withdask:
            self.client.close()

    def run_create_zarr_hierarchy(self, hdf5obj):
        """  Scan hdf5 file and recursively create zarr attributes, groups and dataset structures for accessing data
        Args:
          hdf5obj: h5py.Group, h5py.File, or h5py.Dataset object where information is gathered from
        """

        if h5py.version.hdf5_version_tuple < (1, 10, 5):
            raise Exception(("HDF5Zarr requires h5py installed with minimum hdf5 version of 1.10.5,\n"
                             f"Current hdf5 version {h5py.version.hdf5_version},\n"
                             "h5py installation: https://h5py.readthedocs.io/en/stable/build.html#custom-installation"))

        self._address_dict = self.store[reference_key] if reference_key in self.store else self._address_dict
        address_dict = self._address_dict
        _filename = self.filename
        _file = self.file
        _client = self.client

        # get cache info
        cache_location = None
        if isinstance(self.filename, fsspec.spec.AbstractBufferedFile):
            try:
                cache_location = (self.filename.cache.location,
                                  self.filename.cache.name,
                                  self.filename.cache.size,
                                  self.filename.cache.fetcher,
                                  self.filename.cache.blocks,
                                  self.filename.cache.blocksize,
                                  self.filename.fs)
            except:
                pass

        # if filename is opened with fsspec, use blocksize argument for cache
        cache_blocksize = None
        if isinstance(self.filename, fsspec.spec.AbstractBufferedFile):
            try:
                cache_blocksize = self.filename.cache.blocksize
                self.filename.cache.blocksize = self.blocksize
            except:
                pass

        def _gethfile(fname):
            nonlocal cache_location
            return HDF5Zarr._gethfile(fname, cache_location=cache_location)

        def get_hdf5obj_nametype(filename, objname: str):
            nonlocal _gethfile
            fileid = _gethfile(filename).id
            if objname != '/':
                objname = bytes(objname, encoding='utf-8')
                link_info = fileid.links.get_info(objname)
                if link_info.type == h5py.h5l.TYPE_SOFT:
                    objname = fileid.links.get_val(objname)
                    while fileid.links.get_info(objname).type == h5py.h5l.TYPE_SOFT:
                        objname = fileid.links.get_val(objname)
                objinfo = h5py.h5g.get_objinfo(fileid, name=objname, follow_link=False)
                objname = objname.decode('utf-8')
                objname = str('/'/PurePosixPath(objname))
            else:
                objinfo = h5py.h5g.get_objinfo(fileid, name=bytes(objname, encoding='utf-8'), follow_link=False)
                link_info = types.SimpleNamespace()
                link_info.type = None
                link_info.u = None
            return objname, objinfo.objno, objinfo.type, link_info.type, link_info.u

        _get_hdf5obj_nametype = _client.submit(get_hdf5obj_nametype, _filename, hdf5obj)

        userblock_size = self.file.userblock_size
        def _get_sizes(_filename):
            # returns file's unit_address_size, unit_length_size, userblock_size
            nonlocal _gethfile, userblock_size
            hf = _gethfile(_filename)
            return *hf.id.get_create_plist().get_sizes(), userblock_size

        _sizes = _client.submit(_get_sizes, _filename)

        uri, _hdf5_regfilters_subset = self.uri, self._hdf5_regfilters_subset
        len_iterchunks = 40
        max_chunksize = self.max_chunksize
        def _createiter(name, link_info, basename=b'', last=None, _filename=None,
                        out=[[],[]], link_info_type_u=None, fid=None, skip_paths=None):
            """
            basename: prefix for object name
            last: check if last item in group to end iterate loop
            """
            nonlocal _gethfile

            if link_info:
                link_info_type, link_info_u = link_info.type, link_info.u
            else:
                link_info_type, link_info_u = link_info_type_u
                if link_info_type_u == (None, None):
                    # occurs when hdf5obj passed points to '/'
                    link_info_type = h5py.h5l.TYPE_HARD

            if _filename is not None and fid is None:
               fid = _gethfile(_filename).id

            basename = b'' if basename == b'/' else basename
            name = basename+b'/'+name
            if name == last:
                return
            if len([n for n in skip_paths if name.startswith(n)]) == 0:
                # only run if name is not already processed
                addr_info = None
                if link_info_type == h5py.h5l.TYPE_EXTERNAL:
                    raise Exception(f"Object {name} is not processed: External Link")
                elif link_info_type == h5py.h5l.TYPE_SOFT:
                    objinfo = h5py.h5g.get_objinfo(fid, name=name)
                    targetpath = HDF5Zarr._get_name(None, name, hfid = fid)
                    #address_dict[objno[0]] = targetpath
                    out[1].append([b'SoftLink', [name, objinfo.objno, objinfo.type, targetpath]])
                elif link_info_type == h5py.h5l.TYPE_HARD:
                    objinfo = h5py.h5g.get_objinfo(fid, name=name, follow_link=False)
                    if link_info_u is None:
                        raise Exception("link_info address is None")

                    if objinfo.type == h5py.h5g.GROUP:
                        zobj = zarr.open(None, mode='a', path=name.decode('utf-8'))
                        group_call_items = [partial(_createiter, basename=name, _filename = _filename),
                                            partial(_createiter, basename=basename, last=name, _filename=_filename),
                                            name]

                    elif objinfo.type == h5py.h5g.DATASET:
                        name = name[:-1] if name.endswith(b'/') else name
                        dsetid = h5py.h5o.open(fid, name)
                        zobj, info = HDF5Zarr._create_zarr_dset(fid, uri, _hdf5_regfilters_subset, dsetid, max_chunksize, name.decode('utf-8'), len_iterchunks)
                        if info is not None:
                            out[1].append([b'Chunks', info])

                    name = name[:-1] if name != b'/' and name.endswith(b'/') else name
                    addr_info = {link_info_u: name.decode('utf-8')}
                    out[1].append((b'Attribute',name,)) # attrs
                    out[0].append([zobj.store, addr_info]) # dict_update
                    if objinfo.type == h5py.h5g.GROUP:
                        return group_call_items


        # allows parallel branching at each sub group when calling id.links.iterate
        def _linkitercall(_createiterfunc, oname, objtype_dset = False, skip_paths = None):
            nonlocal _gethfile, _filename
            idx = 0
            fid = _gethfile(_filename).id
            funccalls = []
            ret = [[],[]]
            while True:
                _createiterfunc = partial(_createiterfunc, out=ret, fid=fid, skip_paths=skip_paths)
                if objtype_dset:
                    valiterate = _createiterfunc(bytes('', encoding='utf-8'), None)
                    break # ret points to output
                valiterate = fid.links.iterate(_createiterfunc, obj_name=oname, info=True, idx=idx)
                if valiterate[0] is None:
                    break
                idx = valiterate[1] - 1
                _partialcreateiter, _createiterfunc, _name = *valiterate[0],
                if len([n for n in skip_paths if _name.startswith(n)]) == 0:
                    funccalls.append([_partialcreateiter, _name])

            return funccalls, ret

        softlinks = []
        futures = []
        targetpath_processed = []
        processed_paths = []
        skip_paths = []
        hdf5obj_targetname, objno, objtype, link_info_type, link_info_u = _get_hdf5obj_nametype.result()
        targetpath_items = [(hdf5obj, hdf5obj_targetname, objno, objtype, link_info_type, link_info_u)]
        while len(targetpath_items) > 0:
            # soft link targets not located in hdf5obj group passed to HDF5Zarr are processed via this loop
            # except first loop, all link_info_type are softlinks
            if len(targetpath_processed) != 0:
                targetpath_i = targetpath_items.pop()
                hdf5obj, hdf5obj_targetname, objno, objtype, link_info_type, link_info_u = targetpath_i

            targetpath_processed.append([hdf5obj_targetname, objno, objtype])

            if link_info_type == h5py.h5l.TYPE_SOFT:
                if objtype != h5py.h5g.DATASET:
                    _partialcreateiter = partial(_createiter, basename=bytes(hdf5obj_targetname, encoding='utf-8'), _filename = _filename)
                    futures += [_client.submit(_linkitercall, _partialcreateiter, bytes(hdf5obj, encoding='utf-8'),
                                               skip_paths = skip_paths)]
                softlinks.insert(0, [bytes(hdf5obj, encoding='utf-8'), objno, objtype, hdf5obj_targetname])
            elif link_info_type == h5py.h5l.TYPE_HARD or (link_info_type, link_info_u) == (None, None):
                if objtype != h5py.h5g.DATASET:
                    _partialcreateiter = partial(_createiter, basename=bytes(hdf5obj_targetname, encoding='utf-8'), _filename = _filename)
                    futures += [_client.submit(_linkitercall, _partialcreateiter, bytes(hdf5obj, encoding='utf-8'),
                                               skip_paths = skip_paths)]
            elif link_info_type == h5py.h5l.TYPE_EXTERNAL:
                raise Exception(f"Passed object {hdf5obj_targetname} is not processed: External Link")
            else:
                raise TypeError

            if objtype == h5py.h5g.DATASET:
                _partialcreateiter = partial(_createiter, basename=bytes(hdf5obj_targetname, encoding='utf-8'),
                                          _filename = _filename, link_info_type_u = (link_info_type, link_info_u))
                futures += [_client.submit(_linkitercall, _partialcreateiter, b'', objtype_dset=True,
                                           skip_paths = skip_paths)]

            if objno[0] not in address_dict:
                address_dict[objno[0]]=hdf5obj_targetname

            if self.collectattrs:
                futures += [_client.submit(HDF5Zarr.copy_attrs_data_to_zarr_store, _filename,
                                           address_dict, bytes(hdf5obj_targetname, encoding='utf-8'),
                                           cache_location)]

            outattrs = []
            dict_update = []
            chunk_futures = []
            seq = as_completed(futures, with_results=True)
            _task_numlim = 40 # TO DO, soft cap on number of tasks
            batches_remaining = []
            for batch in seq.batches():
                count_remaining = seq.count() # counts both in progress and finished

                # process batches that do not call new tasks
                keep_batch = []
                for batch_item in batch:
                    future, results = batch_item
                    # check if copy_attrs_data_to_zarr_store call
                    if isinstance(results, dict):
                        outattrs.append(results)
                    elif results is None:
                        # check if dset object
                        if objtype != h5py.h5g.DATASET:
                            raise Exception(f'None return only in case of hdf5_group referring to a dataset')
                    else:
                        keep_batch.append(batch_item)
                batch = keep_batch

                # update batch_process
                numbatchitems = _task_numlim - count_remaining
                if numbatchitems > 0:
                    batch_process = batches_remaining[:numbatchitems]
                    batches_remaining[:numbatchitems] = []
                    numbatchitems = numbatchitems - len(batch_process)
                    if numbatchitems > 0:
                        batch_process += batch[:numbatchitems]
                        batch[:numbatchitems] = []
                    batches_remaining += batch
                else:
                    batches_remaining += batch
                    continue

                for future, results in batch_process:
                    # check if copy_attrs_data_to_zarr_store call
                    if isinstance(results, dict):
                        outattrs.append(results)
                        continue
                    elif results is None:
                        # check if dset object
                        if objtype != h5py.h5g.DATASET:
                            raise Exception(f'None return only in case of hdf5_group referring to a dataset')
                        continue

                    # linkitercall
                    funccalls, outitem = results
                    for call_item in funccalls:
                        item_name = call_item[1]
                        if len([n for n in skip_paths if item_name.startswith(n)]) == 0:
                            seq.add(_client.submit(_linkitercall, *call_item, skip_paths = skip_paths, pure=False))
                            if self.collectattrs:
                                seq.add(_client.submit(HDF5Zarr.copy_attrs_data_to_zarr_store, _filename, address_dict,
                                                       item_name, cache_location))

                    dict_update += outitem[0]
                    for rettype, item in outitem[1]:
                        if rettype == b'SoftLink':
                            # soft links
                            softlinks.append(item)
                        elif rettype == b'Attribute':
                            if self.collectattrs:
                                item_name = item
                                seq.add(_client.submit(HDF5Zarr.copy_attrs_data_to_zarr_store, _filename, address_dict,
                                                       item_name, cache_location))
                        elif rettype == b'Chunks':
                            # chunked
                            # TO DO
                            item_name, dset_chunks, _chunks, chunk_maxind = item
                            chunk_indices = np.indices(chunk_maxind)\
                                              .transpose(*range(1, len(chunk_maxind)+1), 0)\
                                              .reshape(np.prod(chunk_maxind), len(chunk_maxind))
                            chunk_indices = [tuple(c) for c in chunk_indices]

                            len_iterchunks = 40
                            max_numcalls = 8
                            if len(chunk_indices)/max_numcalls > len_iterchunks:
                                len_iterchunks = len(chunk_indices)//max_numcalls + 1
                            numcalls = len(chunk_indices)//len_iterchunks + 1
                            for c in range(numcalls):
                                _indices = chunk_indices[c*len_iterchunks:(c+1)*len_iterchunks]
                                if len(_indices) > 0:
                                    _chfutureattr = self.client.submit(HDF5Zarr._get_storage_info_chunked, dset_chunks, _chunks,
                                                                       chunk_maxind=chunk_maxind, chunk_indices=_indices,
                                                                       name=bytes(item_name, 'utf-8'), filename=_filename,
                                                                       cache_location=cache_location)
                                    chunk_futures.append(_chfutureattr)

            seq = as_completed(chunk_futures, with_results=True)
            for future, results in seq:
                if isinstance(results, tuple):
                    outattrs.append(results)
                else:
                    raise Exception("Expected tuple for chunk futures")

            if len(targetpath_processed) == 1:
                # collect references only if not all are already in address_dict
                if self.collectrefs and hdf5obj_targetname != '/':
                    if self.collectrefs is True:
                        def _get_address(name, info):
                            obj = self.file[name]
                            address_dict[info.addr] = obj.name

                        # add object addresses in file to address_dict
                        address_dict[h5py.h5o.get_info(self.file.id).addr] = self.file.name
                        h5py.h5o.visit(self.file.id, _get_address, obj_name=bytes(self.file.name, encoding='utf-8'), info=True)
                    else:
                        self.collectrefs = self.collectrefs if not isinstance(self.collectrefs, str) else [self.collectrefs]
                        for name in self.collectrefs:
                            name = str("/"/PurePosixPath(name))  # get absolute name
                            if name in address_dict.values():
                                continue
                            link_info = self.file.id.links.get_info(bytes(name, encoding='utf-8'))
                            if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                                warn(f"Skipped {name} in collectrefs, External Link")
                                continue
                            if link_info.type == h5py.h5l.TYPE_SOFT:
                                try:
                                    name = self.get_name(self.file, name)
                                    address_dict[link_info.u] = name
                                except:
                                    warn(f"Skipped {name} in collectrefs, Soft Link. target address not found")
                            else:  # hard link
                                address_dict[link_info.u] = name

                unit_address_size, unit_length_size, userblock_size = _sizes.result()
                address_dict['source'] = {'uri': uri,
                                          'offset_byte_size': unit_address_size,
                                          'length_byte_size': unit_length_size,
                                          'userblock_size': userblock_size}

            zgroup = zarr.open(self.store, mode='a')
            for store_update, addr_info in dict_update:
                for key, val in store_update.items():
                    if key in self.store and val != self.store[key]:
                        info = {key: val}
                        name = str('/'/PurePosixPath(key).parent)
                        info['source'] = {'uri': uri,
                                          'array_name': name}
                        zarray = zgroup[name]
                        FileChunkStore._chunks_info_update(zarray, info)
                    else:
                        self.store[key] = val
                for key, val in addr_info.items():
                    if key in address_dict:
                        if val != address_dict[key]:
                            raise Exception(f'key {key}, address information, already in store with different value')
                    else:
                        address_dict[key] = val

            for item in outattrs:
                if isinstance(item, tuple):
                    info = item[0]
                    name = item[1].decode('utf-8')
                    info['source'] = {'uri': uri,
                                      'array_name': name}
                    zarray = zgroup[name]
                    FileChunkStore._chunks_info_update(zarray, info)
                else:
                    for key, val in item.items():
                        if key.endswith(attrs_key):
                            if key in self.store:
                                if val != self.store[key]:
                                    valb = val.decode('utf-8') if isinstance(val, bytes) else val
                                    if valb != self.store[key]:
                                        info = {key: val}
                                        name = str('/'/PurePosixPath(key).parent)
                                        info['source'] = {'uri': uri,
                                                          'array_name': name}
                                        zarray = zgroup[name]
                                        FileChunkStore._chunks_info_update(zarray, info)
                            else:
                                self.store[key] = val.decode('utf-8') if isinstance(val, bytes) else val

            processed_paths.append(hdf5obj_targetname)
            skip_paths.append(bytes(hdf5obj_targetname, encoding='utf-8'))
            # add soft links
            # loop to collect symlinks, will run only once if hdf5obj is h5py.File or h5py.Dataset
            # if hdf5obj is instance of h5py.Group with symlinks pointing to other groups outside the hdf5obj group,
            # then recursively follows the symlinks
            remattrs = []
            _to_targetpath = dict()
            _to_softlinks = set()
            _to_softlinks_paths = []
            for item in softlinks:
                name_abs, objno, objtype, targetpath = item  # name_abs is softlink's absolute path
                # name_abs: bytes, objno: tuple, targetpath: str
                # get absolute target path
                if objno[0] in address_dict:
                    if targetpath != address_dict[objno[0]]:
                        raise Exception(f"target path {targetpath} for obj {objno[0]} differs from path already " +
                                        f"in store {address_dict[objno[0]]}")
                else:
                    address_dict[objno[0]] = targetpath

                if targetpath not in zgroup or (targetpath == '/' and targetpath not in processed_paths):
                    if self.follow_symlinks:
                        _to_softlinks.add(tuple(item))
                        # targetpath starts with '/'
                        if len([True for n in _to_softlinks_paths if targetpath.startswith(n)]) == 0:
                            print(f"soft link {name_abs.decode('utf-8')} refers to object {targetpath} not in {hdf5obj} group" +
                                  f" creating hierarchy for {targetpath}")
                            _to_softlinks_paths = [n for n in _to_softlinks_paths if not n.startswith(targetpath)]
                            _to_softlinks_paths.append(targetpath)
                            if len([targetpath for n in processed_paths if targetpath.startswith(n)]) == 0:
                                link_info_u = objno[0]
                                targetpath_i = (targetpath, targetpath, objno, objtype, h5py.h5l.TYPE_HARD, link_info_u)
                                _to_targetpath[targetpath] = targetpath_i
                            else:
                                raise Exception("Error in processing softlinks")

                elif name_abs not in zgroup and objtype in (h5py.h5g.DATASET, h5py.h5g.GROUP):
                    # targetpath in self.zgroup
                    name_abs = name_abs.decode('utf-8')
                    ztarget = zgroup[targetpath]
                    # copy store from target
                    HDF5Zarr.copy_storeobj(self.store, targetpath, name_abs, copy_chunks_info = True)
                    if isinstance(ztarget, zarr.Array):
                        zgroup[name_abs].attrs[SYMLINK] = targetpath
                    elif isinstance(ztarget, zarr.Group):
                        zgroup_path = zgroup[name_abs].create_group(SYMLINK, overwrite=True)
                        zgroup_path.attrs[name_abs] = targetpath

            softlinks = list(_to_softlinks)
            targetpath_items = [el for el in targetpath_items if el[0] != '/' and el[0] not in zgroup]
            _to_targetpath = [val for targetpath,val in _to_targetpath.items() if targetpath in _to_softlinks_paths]
            targetpath_items = list(set(targetpath_items + _to_targetpath))
            targetpath_items.sort(key=lambda x: x[1]) # sort according to targetpath

        FileChunkStore.obj_address_info(self.store, address_dict)

        # revert blocksize
        if cache_blocksize is not None:
            try:
                self.filename.cache.blocksize = cache_blocksize
            except:
                pass

    def consolidate_metadata(self, metadata_key='.zmetadata'):
        '''
        Wrapper over zarr.consolidate_metadata to pass chunk store when opening the zarr store
        '''

        # same as zarr.consolidate_metadata(self.store, metadata_key) call,
        # only with key.endswith('.zchunkstore') in is_zarr_key, and passing chunk store
        def is_zarr_key(key):
            return (key.endswith('.zchunkstore') or
                    key.endswith('.zrefstore') or
                    key.endswith('.zarray') or
                    key.endswith('.zgroup') or
                    key.endswith('.zattrs'))

        out = {
            'zarr_consolidated_format': 1,
            'metadata': {
                key: json_loads(self.store[key])
                for key in self.store if is_zarr_key(key)
            }
        }
        self.store[metadata_key] = json_dumps(out)

        meta_store = ConsolidatedMetadataStore(self.store, metadata_key=metadata_key)

        store_mode_cons = 'r' if self.store_mode == 'r' else 'r+'
        self.zgroup = zarr.open(store=meta_store, mode=store_mode_cons,
                                chunk_store=self.zgroup.chunk_store, path=self.store_path)

        return self.zgroup

    @staticmethod
    def get_filename(filename, cache_location = None, _cacheblocks = True):
        if cache_location is not None:
            location, name, size, fetcher, blocks, blocksize, fs = cache_location
            filename = fs.open(filename.path, mode='rb', cache_type='mmap', block_size=blocksize, size=size, cache_options={'location':location, 'blocks': blocks})
            return filename

        if _cacheblocks:
            location = filename.cache.location
            blocks = filename.cache.blocks
            name = filename.cache.name
            size = filename.cache.size
            fetcher = filename.cache.fetcher
            blocksize = filename.cache.blocksize
            fs = fsspec.filesystem('https')
            filename = fs.open(filename.path, mode='rb', cache_type='mmap', block_size=blocksize, size=size, cache_options={'location':location, 'blocks': blocks})
        elif isinstance(filename, str):
            fname = None
            c = 0
            while fname is None:
                # in case too many files are open
                try:
                    fname = local.LocalFileSystem(skip_instance_cache=True).open(filename, mode='rb')
                except Exception as ex:
                    if c > 3600:
                        raise ex
                    else:
                        time.sleep(0.4)
                c += 1
            filename = fname
        else:
            fs = filename.fs
            filename = fs.open(filename.path, mode='rb')
        return filename


    @staticmethod
    def _gethfile(filename, cache_location=None):
        if cache_location is None:
            if isinstance(filename, str):
                fname = HDF5Zarr.get_filename(filename, cache_location, _cacheblocks=False)
                return h5py.File(fname, mode='r')
            hf = h5py.File(filename, mode='r')
            return hf
        hf = h5py.File(HDF5Zarr.get_filename(filename, cache_location), mode='r')
        return hf

    @staticmethod
    def _fill_regfilters(n=None):
        if n is not None:
            return HDF5Zarr._hdf5_regfilters_subset[n]

        # h5py.h5z.FILTER_DEFLATE == 1
        HDF5Zarr._hdf5_regfilters_subset[1] = numcodecs.Zlib

        # h5py.h5z.FILTER_SHUFFLE == 2
        HDF5Zarr._hdf5_regfilters_subset[2] = None

        # h5py.h5z.FILTER_FLETCHER32 == 3
        HDF5Zarr._hdf5_regfilters_subset[3] = None

        # h5py.h5z.FILTER_SZIP == 4
        HDF5Zarr._hdf5_regfilters_subset[4] = None

        # h5py.h5z.FILTER_SCALEOFFSET == 6
        HDF5Zarr._hdf5_regfilters_subset[6] = None

        # LZO
        HDF5Zarr._hdf5_regfilters_subset[305] = None

        # BZIP2
        HDF5Zarr._hdf5_regfilters_subset[307] = numcodecs.BZ2

        # LZF
        HDF5Zarr._hdf5_regfilters_subset[32000] = None

        # Blosc
        HDF5Zarr._hdf5_regfilters_subset[32001] = numcodecs.Blosc

        # Snappy
        HDF5Zarr._hdf5_regfilters_subset[32003] = None

        # LZ4
        HDF5Zarr._hdf5_regfilters_subset[32004] = numcodecs.LZ4

        # bitshuffle
        HDF5Zarr._hdf5_regfilters_subset[32008] = None

        # JPEG-LS
        HDF5Zarr._hdf5_regfilters_subset[32012] = None

        # Zfp
        HDF5Zarr._hdf5_regfilters_subset[32013] = None

        # Fpzip
        HDF5Zarr._hdf5_regfilters_subset[32014] = None

        # Zstandard
        HDF5Zarr._hdf5_regfilters_subset[32015] = numcodecs.Zstd

        # FCIDECOMP
        HDF5Zarr._hdf5_regfilters_subset[32018] = None

    @staticmethod
    def _get_attrval(attrid):
        dtype = attrid.dtype
        shape = attrid.shape
        htype = h5py.h5t.py_create(dtype)
        if dtype.subdtype is not None:
            subdtype, subshape = dtype.subdtype
            shape = attr.shape + subshape   # (5, 3)
            dtype = subdtype                # 'f'
        arr = np.ndarray(shape, dtype=dtype, order='C')
        attrid.read(arr, mtype=htype)

        string_info = h5py.h5t.check_string_dtype(dtype)
        if string_info and (string_info.length is None):
            arr = np.array([
                b.decode('utf-8', 'surrogateescape') for b in arr.flat
            ], dtype=dtype).reshape(arr.shape)
        return arr[()] if len(arr.shape) == 0 else arr


    @staticmethod
    def copy_attrs_data_to_zarr_store(filename, address_dict, name, cache_location=None):
        """ Convert hdf5 attributes to json compatible form and create zarr attributes
        Args:
            h5obj:   hdf5 object
            zobj:    zarr object
        """

        zobj = zarr.open(None, mode='a', path=name)

        fileid = HDF5Zarr._gethfile(filename, cache_location).id

        attrs = dict()
        attrid = True
        n = 0
        while attrid is not None:
            try:
                attrid = h5py.h5a.open(fileid, obj_name=name, index=n)
            except Exception as ex:
                attrid = None
            if attrid is None:
                break

            attrval = HDF5Zarr._get_attrval(attrid)
            key = attrid.get_name()
            attrs[key.decode('utf-8')]=attrval
            n+=1

        for key, val in attrs.items():

            # convert object references in attrs to str
            # e.g. h5py.h5r.Reference instance to "/processing/ophys/ImageSegmentation/ImagingPlane"
            if isinstance(val, str):
                zobj.attrs[key] = val
                continue

            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, np.bool_):
                val = np.bool(val)
            elif isinstance(val, (np.ndarray, np.number)):
                if val.dtype.kind == 'S':
                    val = np.char.decode(val, 'utf-8')
                    val = val.tolist()
                else:
                    val = val.tolist()
            elif isinstance(val, h5py.h5r.Reference):
                if val:
                    # not a null reference
                    try:
                        deref_id = h5py.h5r.dereference(val, fileid)
                        objinfo = h5py.h5g.get_objinfo(deref_id)  # get_objinfo follow_link arg determines target of soft links
                        objno = objinfo.objno
                        if objno[0] in address_dict:
                            deref_objname = address_dict[objno[0]]
                        else:
                            deref_objname = h5py.h5r.get_name(val, fileid)
                            deref_objname = deref_objname.decode('utf-8')
                    except:
                        print(f"Attribute value of type {type(val)} is not processed: \
                                Attribute {key} of object {name}, unable to get target name")
                        continue

                    if deref_objname:
                        val = '//' + deref_objname
                    else:
                        print(f"Attribute value of type {type(val)} is not processed: \
                                Attribute {key} of object {name}, anonymous target")
                        continue
                else:
                    val = None
            elif isinstance(val, h5py.h5r.RegionReference):
                print(f"Attribute value of type {type(val)} is not processed: Attribute {key} of object {name}")
                continue

            attrs[key]=val

        try:
            zobj.attrs.put(attrs)
            return zobj.store
        except Exception:
            # iterate over attributes
            pass

        for key, val in attrs.items():
            try:
                zobj.attrs[key] = val
            except Exception:
                print(f"Attribute value of type {type(val)} is not processed: Attribute {key} of object {name}")

        return zobj.store


    @staticmethod
    def storage_info(dsid, dset_chunks, _chunks, len_iterchunks=None):
        if dsid.shape is None:
            # Null dataset
            return dict()

        if _chunks is None:
            # get offset for Non-External datasets
            if dsid.get_offset() is None:
                return dict()
            else:
                if dset_chunks is None:
                    key = (0,) * (len(dsid.shape) or 1)
                    return {key: {'offset': dsid.get_offset(),
                                  'size': dsid.get_storage_size()}}
                else:
                    stinfo = dict()

                    bytes_offset = dsid.get_offset()
                    storage_size = dsid.get_storage_size()
                    key = (0,)*len(dset_chunks)

                    offsets_, sizes_, chunk_indices = HDF5Zarr._get_chunkstorage_info(dsid, bytes_offset, dsid.shape,
                                                                                  storage_size, dset_chunks, key)

                    for i in range(len(chunk_indices)):
                        stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                         'size': sizes_[i]}

                    return stinfo

        else:
            chunk_maxind = np.ceil([a / b for a, b in zip(dsid.shape, _chunks)]).astype(int)
            if len_iterchunks and np.prod(chunk_maxind)>len_iterchunks:
                return (dset_chunks, _chunks, chunk_maxind)
            else:
                stinfo, _ = HDF5Zarr._get_storage_info_chunked(dset_chunks, _chunks, dsid=dsid, chunk_maxind=chunk_maxind)
                return stinfo

    @staticmethod
    def _get_storage_info_chunked(dset_chunks, _chunks, dsid=None, chunk_maxind=None, chunk_indices=None, name=None,
                                  filename=None, cache_location=None):

        space_status = h5py.h5d.SPACE_STATUS_ALLOCATED
        stinfo = dict()


        if dsid is None:
            # pass name and cache_location
            fileid = HDF5Zarr._gethfile(filename, cache_location).id
            dsid = h5py.h5o.open(fileid, name)

        # TO DO #
        if space_status == h5py.h5d.SPACE_STATUS_ALLOCATED:
            if chunk_indices is None:
                chunk_indices = np.indices(chunk_maxind)\
                                  .transpose(*range(1, len(chunk_maxind)+1), 0)\
                                  .reshape(np.prod(chunk_maxind), len(chunk_maxind))
                chunk_indices = [tuple(c) for c in chunk_indices]

            _indices_chunks = [sum([i[k]*np.prod(chunk_maxind[k+1:]) for k in range(len(chunk_maxind)-1)])+i[-1] for i in chunk_indices]

            blob = None
            for index in _indices_chunks:
                try:
                    blob = dsid.get_chunk_info(index)
                except:
                    blob = None

                if blob is None:
                    break
                bytes_offset = blob.byte_offset
                blob_size = blob.size
                if blob_size is not None and bytes_offset is not None and blob_size>0 and bytes_offset >=0:  # blob.size == 0 when not allocated
                    key = tuple([a // b for a, b in zip(blob.chunk_offset, dset_chunks)])


                    offsets_, sizes_, chunk_indices = HDF5Zarr._get_chunkstorage_info(dsid, bytes_offset, _chunks,
                                                                                  blob_size, dset_chunks, key)

                    for i in range(len(chunk_indices)):
                        stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                         'size': sizes_[i]}

            if blob is not None:
                return stinfo, name

        # get_num_chunks returns the number of all written chunks, regardless of the dataspace.
        # HDF5 1.10.5
        num_chunks = dsid.get_num_chunks()
        if num_chunks == 0:
            return stinfo, name

        _indices_chunks = [i for i in _indices_chunks if index<i<num_chunks]
        for index in _indices_chunks:
            blob = dsid.get_chunk_info(index)

            bytes_offset = blob.byte_offset
            blob_size = blob.size

            key = tuple([a // b for a, b in zip(blob.chunk_offset, dset_chunks)])

            offsets_, sizes_, chunk_indices = HDF5Zarr._get_chunkstorage_info(dsid, bytes_offset, _chunks,
                                                                          blob_size, dset_chunks, key)

            for i in range(len(chunk_indices)):
                stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                 'size': sizes_[i]}

        return stinfo, name

    @staticmethod
    def _get_chunkstorage_info(dset, bytes_offset, blob_shape, blob_size, dset_chunks, key):

        chunk_maxind = np.ceil([a / b for a, b in zip(blob_shape, dset_chunks)]).astype(int)
        chunk_indices = np.indices(chunk_maxind)\
                          .transpose(*range(1, len(chunk_maxind)+1), 0)\
                          .reshape(np.prod(chunk_maxind), len(chunk_maxind))

        strides_ = np.empty(len(chunk_maxind), dtype=int)
        strides_[-1] = dset_chunks[-1]*dset.dtype.itemsize
        for dim_ in range(len(blob_shape)-1):
            strides_[dim_] = dset_chunks[dim_]*np.prod(blob_shape[dim_+1:])*dset.dtype.itemsize
        offsets_ = bytes_offset + np.sum(strides_*chunk_indices, axis=1)
        offsets_ = offsets_.tolist()

        sizes_ = np.empty(len(chunk_indices), dtype=int)
        sizes_[0:-1] = np.diff(offsets_)
        sizes_[-1] = blob_size - (offsets_[-1] - bytes_offset)
        sizes_ = sizes_.tolist()

        chunk_indices = chunk_indices + np.array(key)*chunk_maxind

        return offsets_, sizes_, chunk_indices

    def vlen_storage_info(self, dset, info):
        if len(info) == 0:
            # a null dataset, or no chunk has been written
            # or external dataset
            return dict()

        dsid = dset.id
        if isinstance(self.filename, fsspec.spec.AbstractBufferedFile):
            file_io = fsspec.open(self.filename.details['name'], 'rb').open()
        elif self.file.driver == 'ros3':
            file_io = fsspec.open(self.filename, 'rb').open()
        else:
            file_handle = self.file.id.get_vfd_handle()
            file_io = io.FileIO(file_handle, closefd=False)
        if dset.file.userblock_size != 0:
            # TO DO #
            pass
        fcid = dset.file.id.get_create_plist()
        unit_address_size, unit_length_size = fcid.get_sizes()

        # TO DO #
        dt_vlen = np.dtype([('size', 'uint32'), ('address', f'uint{unit_address_size*8}'), ('id', 'uint32')])
        signature_version_size = 8

        if dset.chunks is None:
            key = (0,) * (len(dset.shape) or 1)
            dsid_string_offset = dsid.get_offset()
            dsid_string_storage_size = dsid.get_storage_size()

            gcol_offsets = self._get_vlenstorage_info(file_io, dsid_string_offset, dsid_string_storage_size, dt_vlen,
                                                      unit_length_size, signature_version_size)

            return {key: {'offset': dsid_string_offset,
                          'size': dsid_string_storage_size,  # size already allocated
                          'gcol_offsets': gcol_offsets}}  # data offsets
        else:
            for key in info:
                if key == 'source' or key == 'array_name':
                    continue

                bytes_offset = info[key]['offset']
                blob_size = info[key]['size']

                # data offsets
                gcol_offsets = self._get_vlenstorage_info(file_io, bytes_offset, blob_size, dt_vlen,
                                                          unit_length_size, signature_version_size)

                info[key].update({'gcol_offsets': gcol_offsets})

            return info

    def _get_vlenstorage_info(self, file_io, dsid_string_offset, dsid_string_storage_size, dt_vlen,
                              unit_length_size, signature_version_size):

        file_io.seek(dsid_string_offset)
        data_ = file_io.read(dsid_string_storage_size)

        data_array = np.frombuffer(data_, dtype=dt_vlen)
        data_offsets = np.unique(data_array['address'])
        data_offsets.sort()
        data_offsets = data_offsets.tolist()

        gcol_offsets = {}
        for offset in data_offsets:
            file_io.seek(offset+signature_version_size)
            size_bytes = file_io.read(unit_length_size)
            gcol_size_i = int.from_bytes(size_bytes, byteorder='little')
            gcol_offsets[offset] = (gcol_size_i,
                                    signature_version_size + unit_length_size)

        return gcol_offsets

    def create_zarr_hierarchy(self, h5py_group, zgroup):
        """  Scan hdf5 file and recursively create zarr attributes, groups and dataset structures for accessing data
        Args:
          h5py_group: h5py.Group or h5py.File object where information is gathered from
          zgroup:     Zarr Group
        """

        # if filename is opened with fsspec, use blocksize argument for cache
        _blocksize = None
        if isinstance(self.filename, fsspec.spec.AbstractBufferedFile):
            try:
                _blocksize = self.filename.cache.blocksize
                self.filename.cache.blocksize = self.blocksize
            except:
                pass

        if isinstance(h5py_group, (h5py.File, h5py.Dataset)):
            h5py_group_name = h5py_group.name
        elif isinstance(h5py_group, h5py.Group):
            h5py_group_name = self.get_name(self.file, h5py_group.name)
        else:
            raise TypeError(f"{h5py_group} should be a h5py.File, h5py.Group or h5py.Dataset")

        if h5py.version.hdf5_version_tuple < (1, 10, 5):
            raise Exception(("HDF5Zarr requires h5py installed with minimum hdf5 version of 1.10.5,\n"
                             f"Current hdf5 version {h5py.version.hdf5_version},\n"
                             "h5py installation: https://h5py.readthedocs.io/en/stable/build.html#custom-installation"))

        self._address_dict = self.store[reference_key] if reference_key in self.store else dict()

        fcid = self.file.id.get_create_plist()
        unit_address_size, unit_length_size = fcid.get_sizes()
        self._address_dict['source'] = {'uri': self.uri,
                                        'offset_byte_size': unit_address_size,
                                        'length_byte_size': unit_length_size,
                                        'userblock_size': self.file.userblock_size}

        def _visit_create_zarr_hierarchy(name, link_info):
            if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                print(f"Object {name} is not processed: External Link")
                return None
            else:
                obj = self.group[name]

                if link_info.type == h5py.h5l.TYPE_HARD:
                    # link_info pointing to hard links stores target address in link_info.u
                    if link_info.u in self._address_dict and self._address_dict[link_info.u] != obj.name:
                        warn("Overwriting object {objname} address present in zarr store")
                    self._address_dict[link_info.u]=obj.name

                # Datasets
                if isinstance(obj, h5py.Dataset):
                    self._create_zarr_hierarchy(obj, self.zgroup)
                # Groups
                elif isinstance(obj, h5py.Group):
                    if obj.name not in self.zgroup or not isinstance(self.zgroup[obj.name], zarr.Group):
                        zgroup_ = self.zgroup.create_group(obj.name, overwrite=True)
                    else:
                        zgroup_ = self.zgroup[obj.name]
                    if link_info.type == h5py.h5l.TYPE_SOFT:
                        zgroup_path = zgroup_.create_group(SYMLINK, overwrite=True)
                        zgroup_path.attrs[obj.name] = self.file.get(obj.name, getlink=True).path

                # attributes
                if self.collectattrs:
                    self._copy_attrs_data_to_zarr_store(obj, self.zgroup[obj.name])

        if not isinstance(h5py_group, h5py.Dataset):
            # add h5py_group address
            targetpath = self.get_name(h5py_group, h5py_group.name)  # get absolute h5py_group name
            objno = h5py.h5g.get_objinfo(h5py_group.id).objno
            self._address_dict[objno[0]]=targetpath

            # create zarr hierarchy
            self.file.id.links.visit(_visit_create_zarr_hierarchy, obj_name=bytes(h5py_group_name, encoding='utf-8'), info=True)
            if self.collectattrs:
                self._copy_attrs_data_to_zarr_store(self.group, self.zgroup)
        else:
            link_info = self.file.id.links.get_info(bytes(self.group.name, encoding='utf-8'))

            if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                raise Exception(f"Dataset {self.group.name} is an External Link")

            if link_info.type == h5py.h5l.TYPE_SOFT:
                warn(f"Dataset {self.group.name} is a Soft Link")

            if self.collectrefs and (link_info.u not in self._address_dict):
                if link_info.type == h5py.h5l.TYPE_SOFT:
                    targetpath = self.get_name(self.file, self.group.name)
                    self._address_dict[link_info.u] = targetpath
                else:  # hard link
                    self._address_dict[link_info.u] = self.group.name
            elif h5py_group.dtype.hasobject and not self.collectrefs:
                warn("Dataset with Object Reference dtypes will dereference to null without collectrefs argument")

            groupname = self.group.parent.name  # dataset parent name is passed to zarr as path
            if groupname in self.zgroup:
                dsetparent = self.zgroup[groupname]
            else:
                dsetparent = self.zgroup.create_group(groupname)
            self._create_zarr_hierarchy(h5py_group, dsetparent)
            self.zgroup = dsetparent[h5py_group.name]
            if self.collectattrs:
                self._copy_attrs_data_to_zarr_store(h5py_group, self.zgroup)

        # collect references only if not all are already in self._address_dict
        if self.collectrefs and h5py_group.name != '/':
            if self.collectrefs is True:
                def _get_address(name, info):
                    obj = self.file[name]
                    self._address_dict[info.addr] = obj.name

                # add object addresses in file to self._address_dict
                self._address_dict[h5py.h5o.get_info(self.file.id).addr] = self.file.name
                h5py.h5o.visit(self.file.id, _get_address, obj_name=bytes(self.file.name, encoding='utf-8'), info=True)
            else:
                self.collectrefs = self.collectrefs if not isinstance(self.collectrefs, str) else [self.collectrefs]
                for name in self.collectrefs:
                    name = str("/"/PurePosixPath(name))  # get absolute name
                    if name in self._address_dict.values():
                        continue
                    link_info = self.file.id.links.get_info(bytes(name, encoding='utf-8'))
                    if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                        warn(f"Skipped {name} in collectrefs, External Link")
                        continue
                    if link_info.type == h5py.h5l.TYPE_SOFT:
                        try:
                            name = self.get_name(self.file, name)
                            self._address_dict[link_info.u] = name
                        except:
                            warn(f"Skipped {name} in collectrefs, Soft Link. target address not found")
                    else:  # hard link
                        self._address_dict[link_info.u] = name

        FileChunkStore.obj_address_info(self.store, self._address_dict)

        # revert blocksize
        if _blocksize is not None:
            try:
                self.filename.cache.blocksize = _blocksize
            except:
                pass

    @staticmethod
    def get_name(hobj, name):
        # used in create_zarr_hierarchy, without dask
        # return a hardlink to name. name is relative to hobj
        if name == '/':
            return name
        linkinfo = hobj.get(name, getlink=True)
        if isinstance(linkinfo, h5py.HardLink):
            return name
        else:
            while True:
                if isinstance(linkinfo, h5py.SoftLink):
                    name = linkinfo.path
                    linkinfo = hobj.file.get(name, getlink=True)
                elif isinstance(linkinfo, h5py.ExternalLink):
                    raise TypeError(f"{name} refers to an External Link. file: {linkinfo.filename}: dataset {linkinfo.path}")
                elif linkinfo is None:
                    raise TypeError(f"{name} is not in {hobj.file}")
                else:
                    break

        return name

    def _create_zarr_hierarchy(self, dset, zgroup):
        """  Scan hdf5 file and recursively create zarr attributes, groups and dataset structures for accessing data
        Args:
          h5py_group: h5py.Group or h5py.File object where information is gathered from
          zgroup:     Zarr Group
        """

        # number of filters
        dcpl = dset.id.get_create_plist()
        nfilters = dcpl.get_nfilters()
        if nfilters > 1:
            # TO DO #
            print(f"Dataset {dset.name} with multiple filters is not processed")
            return None
        elif nfilters == 1:
            # get first filter information
            filter_tuple = dset.id.get_create_plist().get_filter(0)
            filter_code = filter_tuple[0]
            if filter_code in self._hdf5_regfilters_subset and self._hdf5_regfilters_subset[filter_code] is not None:
                # TO DO
                if filter_code == 32001:
                    # Blosc
                    blosc_names = {0: 'blosclz', 1: 'lz4', 2: 'lz4hc', 3: 'snappy', 4: 'zlib', 5: 'zstd'}
                    clevel, shuffle, cname_id = filter_tuple[2][-3:]
                    cname = blosc_names[cname_id]
                    compression = self._hdf5_regfilters_subset[filter_code](cname=cname, clevel=clevel,
                                                                            shuffle=shuffle)
                else:
                    compression = self._hdf5_regfilters_subset[filter_code](level=filter_tuple[2])
            else:
                print(f"Dataset {dset.name} with compression filter {filter_tuple[3]}, hdf5 filter number {filter_tuple[0]} is not processed:\
                        no compatible zarr codec")
                return None
        else:
            compression = None

        object_codec = None
        dtype_refs = None

        if dset.dtype.names is not None:
            # Structured array with Reference dtype

            dset_type = dset.id.get_type()
            dt_nmembers = dset_type.get_nmembers()

            dtype_ = []
            dset_fillvalue = list(dset.fillvalue)
            for dt_i in range(dt_nmembers):
                dtname = dset.dtype.names[dt_i]
                if dset_type.get_member_class(dt_i) == h5py.h5t.REFERENCE:
                    dtype_ += [(dtname, object)]
                    if dset.fillvalue[dt_i]:
                        dset_fillvalue[dt_i] = h5py.h5o.get_info(h5py.h5r.dereference(
                                                                 dset.fillvalue[dt_i], self.file.id)).addr
                    else:
                        dset_fillvalue[dt_i] = 0
                else:
                    dtype_ += [(dtname, dset.dtype.base[dt_i])]

            # currently not using the fill value with structured array
            # containing object, zarr v2.4.1, zarr PR 422
            if all([dt != object for dtname, dt in dtype_]):
                dset_fillvalue = dset.fillvalue
            else:
                dset_fillvalue = None

            dtype_ = np.dtype(dtype_)

            zarray = zgroup.create_dataset(dset.name, shape=dset.shape,
                                           dtype=dtype_,
                                           chunks=dset.chunks or False,
                                           fill_value=dset_fillvalue,
                                           compression=compression,
                                           overwrite=True)

            dtype_refs = [(dset.dtype.names[i],
                           "Object Reference" if dset_type.get_member_class(i) == h5py.h5t.REFERENCE else dset.dtype[i].str)
                          for i in range(dt_nmembers)]
            dtype_refs = dict(dtype=dtype_refs)

            dset_chunks = dset.chunks

        # variable-length Datasets
        elif h5py.check_vlen_dtype(dset.dtype):
            if not h5py.check_string_dtype(dset.dtype):
                print(f"Dataset {dset.name} is not processed: Variable-length dataset, not string")
                return None
            else:
                object_codec = VLenHDF5String()
                zarray = zgroup.create_dataset(dset.name, shape=dset.shape,
                                               dtype=object,
                                               chunks=dset.chunks or False,
                                               fill_value=dset.fillvalue,
                                               compression=compression,
                                               overwrite=True,
                                               object_codec=object_codec)
                dset_chunks = dset.chunks

        elif dset.dtype.hasobject:
            # TO DO test #
            dset_type = dset.id.get_type()

            if dset_type.get_class() == h5py.h5t.REFERENCE:
                dtype_ = np.dtype('|O')

                if dset.fillvalue:
                    dset_fillvalue = h5py.h5o.get_info([h5py.h5r.dereference(dset.fillvalue, self.file.id)]).addr
                else:
                    dset_fillvalue = None

                zarray = zgroup.create_dataset(dset.name, shape=dset.shape,
                                               dtype=dtype_,
                                               chunks=dset.chunks or False,
                                               fill_value=dset_fillvalue,
                                               compression=compression,
                                               overwrite=True,
                                               object_codec=numcodecs.AsType(encode_dtype='|O', decode_dtype='|O'))

                dtype_refs = dict(dtype="Object Reference")

                dset_chunks = dset.chunks

            elif dset_type.get_class() == h5py.h5t.STD_REF_DSETREG:
                print(f"Dataset {dset.name} is not processed: Region Reference dtype")
                return None
            else:
                print(f"Dataset {dset.name} is not processed: Object dtype")
                return None

        else:
            if (self.max_chunksize is not None and compression is None and
                    np.prod(dset.shape) != 0 and (dset.chunks is None or dset.chunks == dset.shape)):

                dset_chunks = dset.chunks if dset.chunks else dset.shape
                if dset.shape != () and dset.size != 0:
                    dset_chunks = list(dset_chunks)
                    dim_ = 0
                    ratio_ = self.max_chunksize/(np.prod(dset_chunks)*dset.dtype.itemsize)
                    while ratio_ < 1:
                        chunk_dim_ = int(ratio_*dset_chunks[dim_])
                        chunk_dim_ = chunk_dim_ if chunk_dim_ else 1
                        chunk_dim_ -= np.argmax(dset_chunks[dim_] % np.arange(chunk_dim_, chunk_dim_//2, -1))
                        dset_chunks[dim_] = int(chunk_dim_)
                        ratio_ = self.max_chunksize/(np.prod(dset_chunks)*dset.dtype.itemsize)
                        dim_ += 1

                    dset_chunks = tuple(dset_chunks)
                dset_chunks = dset_chunks or None
            else:
                dset_chunks = dset.chunks
                if dset_chunks is None and np.prod(dset.shape) == 0:
                    dset_chunks = tuple(s if s != 0 else 1 for s in dset.shape)

            zarray = zgroup.create_dataset(dset.name, shape=dset.shape,
                                           dtype=dset.dtype,
                                           chunks=dset_chunks or False,
                                           fill_value=dset.fillvalue,
                                           compression=compression,
                                           overwrite=True)

        info = self.storage_info(dset.id, dset_chunks, dset.chunks)

        # Store metadata
        if info:
            info['source'] = {'uri': self.uri,
                              'array_name': dset.name}

            if dtype_refs is not None:
                info['source'].update(dtype_refs)
            if object_codec is not None:
                info['source'].update({'type': 'vlen'})
            FileChunkStore.chunks_info(zarray, info)

    def _copy_attrs_data_to_zarr_store(self, h5obj, zobj):
        """ Convert hdf5 attributes to json compatible form and create zarr attributes
        Args:
            h5obj:   hdf5 object
            zobj:    zarr object
        """

        try:
            attrs = dict(h5obj.attrs)
            zobj.attrs.put(attrs)
            return
        except Exception:
            # iterate over attributes
            pass

        for key, val in h5obj.attrs.items():

            # convert object references in attrs to str
            # e.g. h5py.h5r.Reference instance to "/processing/ophys/ImageSegmentation/ImagingPlane"
            if isinstance(val, str):
                zobj.attrs[key] = val
                continue

            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, np.bool_):
                val = np.bool(val)
            elif isinstance(val, (np.ndarray, np.number)):
                if val.dtype.kind == 'S':
                    val = np.char.decode(val, 'utf-8')
                    val = val.tolist()
                else:
                    val = val.tolist()
            elif isinstance(val, h5py.h5r.Reference):
                if val:
                    # not a null reference
                    try:
                        deref_id = h5py.h5r.dereference(val, self.file.id)
                        objinfo = h5py.h5g.get_objinfo(deref_id)  # get_objinfo follow_link arg determines target of soft links
                        objno = objinfo.objno
                        if objno[0] in self._address_dict:
                            deref_objname = self._address_dict[objno[0]]
                        else:
                            deref_objname = h5py.h5r.get_name(val, self.file.id)
                            deref_objname = deref_objname.decode('utf-8')
                    except:
                        print(f"Attribute value of type {type(val)} is not processed: \
                                Attribute {key} of object {h5obj.name}, unable to get target name")
                        continue

                    if deref_objname:
                        val = '//' + deref_objname
                    else:
                        print(f"Attribute value of type {type(val)} is not processed: \
                                Attribute {key} of object {h5obj.name}, anonymous target")
                        continue
                else:
                    val = None
            elif isinstance(val, h5py.h5r.RegionReference):
                print(f"Attribute value of type {type(val)} is not processed: Attribute {key} of object {h5obj.name}")
                continue

            try:
                zobj.attrs[key] = val
            except Exception:
                print(f"Attribute value of type {type(val)} is not processed: Attribute {key} of object {h5obj.name}")

    @staticmethod
    def _get_name(_filename, objname: bytes, hfid = None) -> str:
        # returns string hardlink path of object name if symlink
        if hfid is None:
            hfid = HDF5Zarr._gethfile(_filename).id
        if not (objname in (b'/',b'')):
            link_info = hfid.links.get_info(objname)
            if link_info.type == h5py.h5l.TYPE_SOFT:
                _name = hfid.links.get_val(objname)
                while _name != b'/' and hfid.links.get_info(_name).type == h5py.h5l.TYPE_SOFT:
                    _name = hfid.links.get_val(_name)
            else:
                _name = objname
        else:
            _name = objname

        _name = _name.decode('utf-8')
        _name = _name if _name.startswith('/') else '/'+_name
        return _name

    @staticmethod
    def copy_storeobj(store, x, y, copy_chunks_info = False):
        """ copy store from location x to y
        ----------
        store : MutableMapping
            Store for hdf5 object address information
        x     : address to copy from, absolute
        y     : valid target location, absolute
        copy_chunks_info: copy chunks_meta_key from array, default False
        """
        x = PurePosixPath(normalize_storage_path(x))
        y = PurePosixPath(normalize_storage_path(y))

        # determine if copying array or group
        if str(x/array_meta_key) in store:
            meta_key = array_meta_key
        elif str(x/group_meta_key) in store:
            meta_key = group_meta_key
        else:
            raise KeyError(f"address {x} does not exist")
        if str(y/array_meta_key) in store or str(y/group_meta_key) in store:
            warn(f"overwriting target address {str(y)} already in zarr store")
        # copy object
        store[str(y/meta_key)]=store[str(x/meta_key)]
        # copy attributes
        xattrkey = str(x/attrs_key)
        if xattrkey in store:
            store[str(y/attrs_key)]=store[xattrkey]

        if copy_chunks_info and meta_key == array_meta_key:
            zchunksx = store[str(x/chunks_meta_key)]
            zchunksx = FileChunkStore._ensure_dict(zchunksx)
            store_item = {str(y/PurePosixPath(k).name) if k != 'source' else k: v for k, v in zchunksx.items()}

            store[str(y/chunks_meta_key)] = json_dumps(store_item)

    @staticmethod
    def _create_zarr_dset(fid, uri, _hdf5_regfilters_subset, dsid, max_chunksize, name, len_iterchunks):
        """  Scan hdf5 file and recursively create zarr attributes, groups and dataset structures for accessing data
        """

        zgroup = zarr.open(None, mode='a',path='/')
        name = str('/'/PurePosixPath(normalize_storage_path(name)))
        # number of filters
        dcpl = dsid.get_create_plist()
        nfilters = dcpl.get_nfilters()
        if nfilters > 1:
            # TO DO #
            print(f"Dataset {name} with multiple filters is not processed")
            return None
        elif nfilters == 1:
            # get first filter information
            filter_tuple = dcpl.get_filter(0)
            filter_code = filter_tuple[0]
            if filter_code in _hdf5_regfilters_subset and _hdf5_regfilters_subset[filter_code] is not None:
                # TO DO
                if filter_code == 32001:
                    # Blosc
                    blosc_names = {0: 'blosclz', 1: 'lz4', 2: 'lz4hc', 3: 'snappy', 4: 'zlib', 5: 'zstd'}
                    clevel, shuffle, cname_id = filter_tuple[2][-3:]
                    cname = blosc_names[cname_id]
                    compression = _hdf5_regfilters_subset[filter_code](cname=cname, clevel=clevel,
                                                                            shuffle=shuffle)
                else:
                    compression = _hdf5_regfilters_subset[filter_code](level=filter_tuple[2])
            else:
                print(f"Dataset {name} with compression filter {filter_tuple[3]}, hdf5 filter number {filter_tuple[0]} is not processed:\
                        no compatible zarr codec")
                return None
        else:
            compression = None

        object_codec = None
        dtype_refs = None


        try:
            chunks = dcpl.get_chunk()
        except:
            chunks = None

        fillvalue = np.ndarray(shape=(1,), dtype = dsid.dtype)
        dcpl.get_fill_value(fillvalue)
        fillvalue = fillvalue[0]
        if dsid.dtype.names is not None:
            # Structured array with Reference dtype
            dset_type = dsid.get_type()
            dt_nmembers = dset_type.get_nmembers()

            dtype_ = []
            dset_fillvalue = list(fillvalue)
            for dt_i in range(dt_nmembers):
                dtname = dsid.dtype.names[dt_i]
                if dset_type.get_member_class(dt_i) == h5py.h5t.REFERENCE:
                    dtype_ += [(dtname, object)]
                    if fillvalue[dt_i]:
                        dset_fillvalue[dt_i] = h5py.h5o.get_info(h5py.h5r.dereference(
                                                                 fillvalue[dt_i], fid)).addr
                    else:
                        dset_fillvalue[dt_i] = 0
                else:
                    dtype_ += [(dtname, dsid.dtype.base[dt_i])]

            # currently not using the fill value with structured array
            # containing object, zarr v2.4.1, zarr PR 422
            if all([dt != object for dtname, dt in dtype_]):
                dset_fillvalue = fillvalue
            else:
                dset_fillvalue = None

            dtype_ = np.dtype(dtype_)

            zarray = zgroup.create_dataset(name, shape=dsid.shape,
                                           dtype=dtype_,
                                           chunks=chunks or False,
                                           fill_value=dset_fillvalue,
                                           compression=compression,
                                           overwrite=True)

            dtype_refs = [(dsid.dtype.names[i],
                           "Object Reference" if dset_type.get_member_class(i) == h5py.h5t.REFERENCE else dsid.dtype[i].str)
                          for i in range(dt_nmembers)]
            dtype_refs = dict(dtype=dtype_refs)

            dset_chunks = chunks

        # variable-length Datasets
        elif h5py.check_vlen_dtype(dsid.dtype):
            if not h5py.check_string_dtype(dsid.dtype):
                print(f"Dataset {name} is not processed: Variable-length dataset, not string")
                return None
            else:
                object_codec = VLenHDF5String()
                zarray = zgroup.create_dataset(name, shape=dsid.shape,
                                               dtype=object,
                                               chunks=chunks or False,
                                               fill_value=fillvalue,
                                               compression=compression,
                                               overwrite=True,
                                               object_codec=object_codec)
                dset_chunks = chunks

        elif dsid.dtype.hasobject:
            # TO DO test #
            dset_type = dsid.get_type()

            if dset_type.get_class() == h5py.h5t.REFERENCE:
                dtype_ = np.dtype('|O')

                if fillvalue:
                    dset_fillvalue = h5py.h5o.get_info([h5py.h5r.dereference(fillvalue, fid)]).addr
                else:
                    dset_fillvalue = None

                zarray = zgroup.create_dataset(name, shape=dsid.shape,
                                               dtype=dtype_,
                                               chunks=chunks or False,
                                               fill_value=dset_fillvalue,
                                               compression=compression,
                                               overwrite=True,
                                               object_codec=numcodecs.AsType(encode_dtype='|O', decode_dtype='|O'))

                dtype_refs = dict(dtype="Object Reference")

                dset_chunks = chunks

            elif dset_type.get_class() == h5py.h5t.STD_REF_DSETREG:
                print(f"Dataset {name} is not processed: Region Reference dtype")
                return None
            else:
                print(f"Dataset {name} is not processed: Object dtype")
                return None

        else:
            if (max_chunksize is not None and compression is None and
                    np.prod(dsid.shape) != 0 and (chunks is None or chunks == dsid.shape)):

                dset_chunks = chunks if chunks else dsid.shape
                if dsid.shape != () and np.prod(dsid.shape) != 0:
                    dset_chunks = list(dset_chunks)
                    dim_ = 0
                    ratio_ = max_chunksize/(np.prod(dset_chunks)*dsid.dtype.itemsize)
                    while ratio_ < 1:
                        chunk_dim_ = int(ratio_*dset_chunks[dim_])
                        chunk_dim_ = chunk_dim_ if chunk_dim_ else 1
                        chunk_dim_ -= np.argmax(dset_chunks[dim_] % np.arange(chunk_dim_, chunk_dim_//2, -1))
                        dset_chunks[dim_] = int(chunk_dim_)
                        ratio_ = max_chunksize/(np.prod(dset_chunks)*dsid.dtype.itemsize)
                        dim_ += 1

                    dset_chunks = tuple(dset_chunks)
                dset_chunks = dset_chunks or None
            else:
                dset_chunks = chunks
                if dset_chunks is None and np.prod(dsid.shape) == 0:
                    dset_chunks = tuple(s if s != 0 else 1 for s in dsid.shape)
            zarray = zgroup.create_dataset(name, shape=dsid.shape,
                                           dtype=dsid.dtype,
                                           chunks=dset_chunks or False,
                                           fill_value=fillvalue,
                                           compression=compression,
                                           overwrite=True)

        info = HDF5Zarr.storage_info(dsid, dset_chunks, chunks, len_iterchunks)

        # Store metadata
        if isinstance(info, tuple):
            _info = dict()
            _info['source'] = {'uri': uri,
                              'array_name': name}
            if dtype_refs is not None:
                _info['source'].update(dtype_refs)
            if object_codec is not None:
                _info['source'].update({'type': 'vlen'})
            FileChunkStore.chunks_info(zarray, _info)
            return zgroup, (name,)+info
        if info:
            info['source'] = {'uri': uri,
                              'array_name': name}

            if dtype_refs is not None:
                info['source'].update(dtype_refs)
            if object_codec is not None:
                info['source'].update({'type': 'vlen'})
            FileChunkStore.chunks_info(zarray, info)

        return zgroup, None

    @staticmethod
    def _rewrite_vlen_to_fixed(h5py_group, changed_dsets={}):
        """  Scan hdf5 file or hdf5 group object and recursively convert variable-length string dataset to fixed-length
        Args:
          h5py_group: h5py.Group or h5py.File object
        """

        if (not isinstance(h5py_group, h5py.File) and
            (not issubclass(h5py_group.file.get(h5py_group.name, getclass=True), h5py.Group) or
             not issubclass(h5py_group.file.get(h5py_group.name, getclass=True, getlink=True), h5py.HardLink))):
            raise TypeError(f"{h5py_group} should be a h5py.File or h5py.Group as a h5py.HardLink")

        # iterate through group members
        group_iter = [name for name in h5py_group.keys()]
        for name in group_iter:
            obj = h5py_group[name]

            # get group member's link class
            obj_linkclass = h5py_group.get(name, getclass=True, getlink=True)

            # Datasets
            if issubclass(h5py_group.get(name, getclass=True), h5py.Dataset):
                if issubclass(obj_linkclass, h5py.ExternalLink):
                    print(f"Skipped rewriting variable-length dataset {obj.name}: External Link")
                    continue
                dset = obj

                # variable-length Datasets
                if h5py.check_vlen_dtype(dset.dtype) and h5py.check_string_dtype(dset.dtype):

                    vlen_stringarr = dset[()]
                    if dset.shape == ():
                        string_lengths_ = len(vlen_stringarr)
                        length_max = string_lengths_
                    else:
                        length_max = max(len(el) for el in vlen_stringarr.flatten())
                    if dset.fillvalue is not None:
                        length_max = max(length_max, len(dset.fillvalue))
                    length_max = length_max + (-length_max) % 8
                    dt_fixedlen = f'|S{length_max}'

                    if isinstance(dset.fillvalue, str):
                        dset_fillvalue = dset.fillvalue.encode('utf-8')
                    else:
                        dset_fillvalue = dset.fillvalue

                    affix_ = '_fixedlen~'
                    dset_name = dset.name
                    h5py_group.file.move(dset_name, dset_name+affix_)
                    changed_dsets[dset_name+affix_] = dset_name
                    dsetf = h5py_group.file.create_dataset_like(dset_name, dset, dtype=dt_fixedlen, fillvalue=dset_fillvalue)

                    # TO DO, copy attrs after all string dataset are moved
                    for key, val in dset.attrs.items():
                        if isinstance(val, (bytes, np.bool_, str, int, float, np.number)):
                            dsetf.attrs[key] = val
                        else:
                            # TO DO #
                            print(f"Moving variable-length string Datasets: attribute value of type\
                                    {type(val)} is not processed. Attribute {key} of object {dsetf.name}")

                    if dsetf.shape == ():
                        if isinstance(vlen_stringarr, bytes):
                            dsetf[...] = vlen_stringarr
                        else:
                            dsetf[...] = vlen_stringarr.encode('utf-8')
                    else:
                        dsetf[...] = vlen_stringarr.astype(dt_fixedlen)

            # Groups
            elif (issubclass(h5py_group.get(name, getclass=True), h5py.Group) and
                  not issubclass(obj_linkclass, h5py.SoftLink)):
                if issubclass(obj_linkclass, h5py.ExternalLink):
                    print(f"Group {obj.name} is not processed: External Link")
                    continue
                changed_dsets = HDF5Zarr._rewrite_vlen_to_fixed(obj, changed_dsets)

        return changed_dsets


# from zarr.storage: #
def _path_to_prefix(path):
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix


chunks_meta_key = '.zchunkstore'
reference_key = '.zrefstore'

class FileChunkStore(MutableMapping):
    """A file as a chunk store.
    Zarr array chunks are all in a single file.
    Parameters
    ----------
    store : MutableMapping
        Store for file chunk location metadata.
    chunk_source : file-like object
        Source (file) containing chunk bytes. Must be seekable and readable.
    """

    def __init__(self, store, chunk_source):
        self._store = store
        if not (chunk_source.seekable and chunk_source.readable):
            raise TypeError(f'{chunk_source}: chunk source is not '
                            'seekable and readable')
        self._source = chunk_source
        self._gcol = {}
        # TO DO #
        self.dt_vlen = np.dtype([('size', 'uint32'), ('address', 'uint64'), ('id', 'uint32')])

    @property
    def store(self):
        """MutableMapping store for file chunk information"""
        return self._store

    @store.setter
    def store(self, new_store):
        """Set the new store for file chunk location metadata."""
        self._store = new_store

    @property
    def source(self):
        """The file object where chunks are stored."""
        return self._source

    @staticmethod
    def _chunks_info_update(zarray, chunks_loc):
        """Store chunks location information for a Zarr array.
        Parameters
        ----------
        zarray : zarr.core.Array
            Zarr array that will use the chunk data.
        chunks_loc : dict
            File storage information for the chunks belonging to the Zarr array.
        """
        if 'source' not in chunks_loc:
            raise ValueError('Chunk source information missing')
        if any([k not in chunks_loc['source'] for k in ('uri', 'array_name')]):
            raise ValueError(
                f'{chunks_loc["source"]}: Chunk source information incomplete')

        key = _path_to_prefix(zarray.path) + chunks_meta_key
        chunks_meta = dict()
        for k, v in chunks_loc.items():
            if k != 'source':
                k = zarray._chunk_key(k)
                if any([a not in v for a in ('offset', 'size')]):
                    raise ValueError(
                        f'{k}: Incomplete chunk location information')
            chunks_meta[k] = v

        # Store Zarr array chunk location metadata...
        if key in zarray.store:
            _chunkinfo = json_loads(zarray.store[key].decode('utf-8'))
            _source = chunks_meta.pop('source')
            if _source['array_name'] != _chunkinfo['source']['array_name'] or _source['uri'] != _chunkinfo['source']['uri']:
                raise Exception
            if len(_source) != 2:
                raise Exception
            _chunkinfo.update(chunks_meta)
            zarray.store[key] = json_dumps(_chunkinfo)
        else:
            zarray.store[key] = json_dumps(chunks_meta)

    @staticmethod
    def chunks_info(zarray, chunks_loc):
        """Store chunks location information for a Zarr array.
        Parameters
        ----------
        zarray : zarr.core.Array
            Zarr array that will use the chunk data.
        chunks_loc : dict
            File storage information for the chunks belonging to the Zarr array.
        """
        if 'source' not in chunks_loc:
            raise ValueError('Chunk source information missing')
        if any([k not in chunks_loc['source'] for k in ('uri', 'array_name')]):
            raise ValueError(
                f'{chunks_loc["source"]}: Chunk source information incomplete')

        key = _path_to_prefix(zarray.path) + chunks_meta_key
        chunks_meta = dict()
        for k, v in chunks_loc.items():
            if k != 'source':
                k = zarray._chunk_key(k)
                if any([a not in v for a in ('offset', 'size')]):
                    raise ValueError(
                        f'{k}: Incomplete chunk location information')
            chunks_meta[k] = v

        # Store Zarr array chunk location metadata...
        zarray.store[key] = json_dumps(chunks_meta)

    @staticmethod
    def obj_address_info(store, address_loc, obj_name=None):
        """Store hdf5 object address information for arrays of references.
        Parameters
        ----------
        store : MutableMapping
            Store for hdf5 object address information
        address_loc : dict
            Dictionary with object addresses as keys and object names as values
        obj_name: str
            path of zarr object in store, default None
        """
        if 'source' not in address_loc:
            raise ValueError('Object reference source information missing')
        if any([k not in address_loc['source'] for k in ('uri', 'offset_byte_size', 'length_byte_size', 'userblock_size')]):
            raise ValueError(
                f'{address_loc["source"]}: Object reference source information incomplete')

        if obj_name is None:
            key = reference_key
        else:
            key = PurePosixPath(obj_name) / reference_key
            key = key.relative_to(key.root)

        if any(not isinstance(k, str) for k in address_loc.keys()):
            address_loc = {str(k): v for k, v in address_loc.items()}

        store[key] = json_dumps(address_loc)

    def _get_chunkstore_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / chunks_meta_key)

    @staticmethod
    def _ensure_dict(obj):
        if isinstance(obj, bytes):
            return json_loads(obj)
        else:
            return obj

    def __getitem__(self, chunk_key):
        """Read in chunk bytes.
        Parameters
        ----------
        chunk_key : str
            Zarr array chunk key.
        Returns
        -------
        bytes
            Bytes of the requested chunk.
        """

        zchunk_key = self._get_chunkstore_key(chunk_key)
        try:
            zchunks = self._ensure_dict(self._store[zchunk_key])
        except KeyError:
            raise KeyError(zchunk_key)

        # get chunk location
        try:
            chunk_loc = zchunks[chunk_key]
        except KeyError:
            raise KeyError(chunk_key)

        # Read chunk's data
        self._source.seek(chunk_loc['offset'], os.SEEK_SET)
        bytes = self._source.read(chunk_loc['size'])

        # variable-length string
        if 'gcol_offsets' in chunk_loc or zchunks['source'].get('type') == 'vlen':
            data_array = np.frombuffer(bytes, dtype=self.dt_vlen)
            data_offsets = np.unique(data_array['address'])
            data_offsets.sort()
            data_offsets = data_offsets.tolist()

            data_bytes = np.empty(shape=len(data_offsets)+1, dtype=object)
            data_bytes[0] = bytes

            length_byte_size = offset_byte_size = userblock_size = None
            signature_version_size = 8
            gcol_offsets = chunk_loc['gcol_offsets'] if 'gcol_offsets' in chunk_loc else dict()

            for i in range(len(data_offsets)):
                offset_item = data_offsets[i]
                if offset_item not in self._gcol:
                    if str(offset_item) not in gcol_offsets:
                        if offset_byte_size is None:
                            ref_key = self._get_reference_key('')
                            address_key = self._ensure_dict(self._store[ref_key])
                            address_key_source = address_key['source']
                            length_byte_size = address_key_source['length_byte_size']
                            offset_byte_size = address_key_source['offset_byte_size']
                            # TO DO userblock_size

                        self._source.seek(offset_item+signature_version_size, os.SEEK_SET)
                        size_bytes = self._source.read(length_byte_size)
                        gcol_size_i = int.from_bytes(size_bytes, byteorder='little')
                        gcol_offsets[str(offset_item)] = (gcol_size_i,
                                                signature_version_size + length_byte_size)

                    gcol_size, skip = gcol_offsets[str(offset_item)]

                    gcol_size = gcol_size - skip
                    offset = offset_item + skip
                    self._source.seek(offset, os.SEEK_SET)
                    gcol_bytes = self._source.read(gcol_size)
                    self._gcol[offset] = gcol_bytes
                    data_bytes[i+1] = gcol_bytes
                else:
                    data_bytes[i+1] = self._gcol[offset]

            return data_bytes

        try:
            # Get array chunk size
            zarray_key = self._get_array_key(chunk_key)
            zarray_key = self._ensure_dict(self._store[zarray_key])
            dtype_str = zarray_key['dtype']
            # compound dtype
            if isinstance(dtype_str, list):
                dtype_str = [tuple(el) for el in dtype_str]

            zarray_itemsize = np.dtype(dtype_str).itemsize
            zarray_chunksize = np.prod(zarray_key['chunks'])*zarray_itemsize
            # Pad up to chunk size
            if len(bytes) < zarray_chunksize:
                bytes = bytes.ljust(zarray_chunksize, b'\0')
        except KeyError:
            raise KeyError(chunk_key)

        # check for object references
        if (dtype_str == '|O' or
           (isinstance(dtype_str, list) and any([dt[1] == '|O' for dt in dtype_str]))):
            try:
                zref_dtype = zchunks['source']['dtype']
                if dtype_str != '|O':
                    zref_dtype = dict(zref_dtype)
            except KeyError:
                raise KeyError(f'{zchunks["source"]}: Chunk source information incomplete, '
                                 '\'dtype\' key missing. Unable to check for Object References')

            # get root reference metadata. Contains hdf5 object addresses
            ref_key = self._get_reference_key('')
            address_key = self._ensure_dict(self._store[ref_key])
            address_key_source = address_key.pop('source')
            address_key = {int(k): v for k, v in address_key.items()}

            if chunk_loc['size'] < zarray_chunksize:
                address_key.setdefault(0, '')

            # function to convert references to string
            # for hanging chunks
            chunkedge = zarray_key['shape'] - (
                zarray_key['chunks']*np.array([int(i) for i in PurePosixPath(chunk_key).name.split('.')])+zarray_key['chunks'])

            ref_array_func = np.frompyfunc(lambda x: address_key[int(x)] if int(x) in address_key else '', 1, 1)
            chunkedge_slice = tuple(slice(None,None if c>=0 else c) for c in chunkedge)
            if dtype_str == '|O':

                if zref_dtype != "Object Reference":
                    raise TypeError
                dtype_ = np.dtype(f'uint{address_key_source["offset_byte_size"]*8}')
                data_array = np.frombuffer(bytes, dtype=dtype_).reshape(zarray_key['chunks'])[chunkedge_slice]
                data_bytes = np.empty(shape=zarray_key['chunks'], dtype=dtype_str)
                data_bytes[chunkedge_slice] = ref_array_func(data_array)
            else:
                # structured array
                try:
                    for dtname, dt in dtype_str:
                        if dtname not in zref_dtype:
                            raise KeyError
                        if dt == '|O':
                            if zref_dtype[dtname] != "Object Reference":
                                raise TypeError
                        else:
                            if dt != zref_dtype[dtname]:
                                raise KeyError
                except KeyError:
                    raise KeyError(f'{zchunks["source"]}: Chunk source information incomplete, '
                                    '\'dtype\' is incompatible with the zarr array dtype')

                dtype_ = [(dtname, dt if dt != '|O' else f'uint{address_key_source["offset_byte_size"]*8}')
                          for dtname, dt in dtype_str]
                dtype_ = np.dtype(dtype_)

                data_array = np.frombuffer(bytes, dtype=dtype_).reshape(zarray_key['chunks'])[chunkedge_slice]
                data_bytes = np.empty(shape=zarray_key['chunks'], dtype=dtype_str)

                for dtname, dt in zref_dtype.items():
                    if dt == "Object Reference":
                        data_bytes[dtname][chunkedge_slice] = ref_array_func(data_array[dtname])
                    else:
                        data_bytes[dtname][chunkedge_slice] = data_array[dtname]
            return data_bytes
        else:
            return bytes

    def _get_array_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / array_meta_key)

    def _get_reference_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / reference_key)

    def _get_attrs_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / attrs_key)

    def __delitem__(self, chunk_key):
        raise RuntimeError(f'{chunk_key}: Cannot delete chunk')

    def keys(self):
        try:
            for key in self._store.keys():
                if key.endswith(chunks_meta_key):
                    chunks_info = self._ensure_dict(self._store[key])
                    for k in chunks_info.keys():
                        if k == 'source':
                            continue
                        yield k
        except AttributeError:
            raise RuntimeError(
                f'{type(self._store)}: Cannot iterate over store keys')

    def __iter__(self):
        return self.keys()

    def __len__(self):
        """Total number of chunks in the file."""
        total = 0
        try:
            for k in self._store.keys():
                if k.endswith(chunks_meta_key):
                    chunks_info = self._ensure_dict(self._store[k])
                    total += (len(chunks_info) - 1)
        except AttributeError:
            raise RuntimeError(
                f'{type(self._store)}: Does not support counting chunks')
        return total

    def __setitem__(self, chunk_key):
        raise RuntimeError(f'{chunk_key}: Cannot modify chunk data')


def rewrite_vlen_to_fixed(filename: str, group: str = None, update_references=False):
    """  Scan hdf5 file or hdf5 group object and recursively convert variable-length string dataset to fixed-length
    Args:
      filename:   str or File-like object, hdf5 file
      group:      str, hdf5 group in hdf5 file to recursively convert variable-lengths strings, default is the root group.
    """

    if group is not None and not isinstance(group, str):
        raise TypeError(f"Expected str for group, recieved {type(group)}")

    with h5py.File(filename, mode='r+') as hfile:
        obj = hfile[group] if group is not None else hfile
        _rewrite_vlen_to_fixed(obj, update_references=update_references)


def _rewrite_vlen_to_fixed(h5py_group, update_references=False):
    """  Scan hdf5 file or hdf5 group object and recursively convert variable-length string dataset to fixed-length
    Args:
      h5py_group: h5py.Group or h5py.File object
    """

    if h5py_group.file.mode != 'r+':
        raise ValueError(f"{h5py_group.file} mode must be 'r+' for rewriting variable-length datasets")

    changed_dsets = HDF5Zarr._rewrite_vlen_to_fixed(h5py_group)

    def _update_references(name, link_info):
        nonlocal changed_dsets, h5py_group

        if link_info.type == h5py.h5l.TYPE_EXTERNAL:
            print(f"Object {name} is not checked for dangling references: External Link")
        elif link_info.type == h5py.h5l.TYPE_SOFT:
            pass
        else:
            obj = h5py_group[name]
            if isinstance(obj, h5py.Dataset):
                dset = obj
                if dset.dtype.names:
                    # TO DO #
                    print(f"Dataset {dset.name} is not checked for dangling references: compound dtype")
                elif h5py.check_ref_dtype(dset.dtype) == h5py.RegionReference:
                    # TO DO #
                    print(f"Dataset {dset.name} is not checked for dangling references: Region Reference dtype")
                elif h5py.check_ref_dtype(dset.dtype) == h5py.Reference:
                    # TO DO #
                    print(f"Dataset {dset.name} is not checked for dangling references")

    if update_references:
        h5py_group.id.links.visit(_update_references, info=True)

    def _update_attr_references(name, link_info):
        nonlocal changed_dsets, h5py_group

        if link_info.type == h5py.h5l.TYPE_SOFT:
            pass
        else:
            obj = h5py_group[name]
            for key, val in obj.attrs.items():
                if isinstance(val, h5py.RegionReference):
                    print(f"Attribute {key} of {obj.name} is not checked for dangling references: Region Reference")
                elif isinstance(val, h5py.Reference):
                    if val:
                        # not a null reference
                        deref_obj = h5py_group.file[val]
                        if deref_obj.name is None:
                            # anonymous dataset
                            pass
                        elif deref_obj.name in changed_dsets:
                            val = changed_dsets[deref_obj.name]
                            obj.attrs[key] = h5py_group.file[val].ref

    h5py_group.id.links.visit(_update_attr_references, info=True)

    for dsetname in changed_dsets:
        del h5py_group.file[dsetname]
