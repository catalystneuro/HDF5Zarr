import h5py
import zarr
from zarr.storage import array_meta_key
from zarr.storage import ConsolidatedMetadataStore
import numpy as np
import numcodecs
import fsspec
from typing import Union
import os
import io
from pathlib import Path
from collections.abc import MutableMapping
from pathlib import PurePosixPath
from zarr.util import json_dumps, json_loads
from xdrlib import Unpacker
import struct
SYMLINK = '.link'


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
            vlen_array[p[i]:p[i+1]] = np.array(vlen_list)[sorter[vlen_str_index]].astype(str)

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


class HDF5Zarr(object):
    """ class to create zarr structure for reading hdf5 files """

    def __init__(self, filename: str, hdf5group: str = None,
                 store: Union[MutableMapping, str, Path] = None, store_path: str = None,
                 store_mode: str = 'a', LRU: bool = False, LRU_max_size: int = 2**30,
                 max_chunksize=2*2**20):

        """
        Args:
            filename:                    str or File-like object, file name string or File-like object to be read by zarr
            hdf5group:                   str, hdf5 group in hdf5 file to be read by zarr
                                         along with its children. default is the root group.
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
                                         'a'          read and write, create if it does not exist, default 'r'
            store_path:                  string, path in zarr store
            LRU:                         bool, if store is not already zarr.LRUStoreCache, add
                                         a zarr.LRUStoreCache store layer on top of currently used store
            LRU_max_size:                int, maximum zarr.LRUStoreCache cache size, only used
                                         if store is zarr.LRUStoreCache, or LRU argument is True
            max_chunksize:               maximum chunk size to use when creating zarr hierarchy, this is useful if
                                         only a small slice of data needs to be read
        """
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

        # store, store_path, and store_mode are passed through to zarr
        self.store_path = store_path
        self.store_mode = store_mode
        if store is not None and LRU is True and not isinstance(store, zarr.LRUStoreCache):
            self.store = zarr.LRUStoreCache(store, max_size=self.LRU_max_size)
        else:
            self.store = store

        # create dictionary mapping hdf5 filter numbers to compatible zarr codec
        self._hdf5_regfilters_subset = {}
        self._fill_regfilters()

        # dictionary to hold addresses of hdf5 objects in file
        self._address_dict = {}

        # create zarr format hierarchy for datasets and attributes compatible with hdf5 file,
        # dataset contents are not copied, unless it contains variable-length strings

        self.zgroup = zarr.open_group(self.store, mode=self.store_mode, path=self.store_path)
        if self.store is None:
            self.store = self.zgroup.store

        # FileChunkStore requires uri
        if isinstance(filename, str):
            self.uri = filename
        else:
            try:
                self.uri = getattr(filename, 'path', None)
                if self.uri is None:
                    self.uri = filename.name
            except Exception:
                self.uri = ''

        # Access hdf5 file and create zarr hierarchy
        if hdf5group is not None and not isinstance(hdf5group, str):
            raise TypeError(f"Expected str for hdf5group, recieved {type(hdf5group)}")
        self.hdf5group = hdf5group
        self.filename = filename
        if self.store_mode != 'r':
            self.file = h5py.File(self.filename, mode='r')
            self.group = self.file[self.hdf5group] if self.hdf5group is not None else self.file
            self.create_zarr_hierarchy(self.group, self.zgroup)
            self.file.close()
        if isinstance(self.filename, str):
            self.chunkstore_file = fsspec.open(self.filename, mode='rb')
            self.chunk_store = FileChunkStore(self.store, chunk_source=self.chunkstore_file.open())
        else:
            self.chunk_store = FileChunkStore(self.store, chunk_source=self.filename)
        if LRU is True and not isinstance(self.chunk_store, zarr.LRUStoreCache):
            self.chunk_store = zarr.LRUStoreCache(self.chunk_store, max_size=self.LRU_max_size)

        # open zarr group
        store_mode_cons = 'r' if self.store_mode == 'r' else 'r+'
        self.zgroup = zarr.open_group(self.store, mode=store_mode_cons, path=self.store_path, chunk_store=self.chunk_store)

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

    def _fill_regfilters(self):

        # h5py.h5z.FILTER_DEFLATE == 1
        self._hdf5_regfilters_subset[1] = numcodecs.GZip

        # h5py.h5z.FILTER_SHUFFLE == 2
        self._hdf5_regfilters_subset[2] = None

        # h5py.h5z.FILTER_FLETCHER32 == 3
        self._hdf5_regfilters_subset[3] = None

        # h5py.h5z.FILTER_SZIP == 4
        self._hdf5_regfilters_subset[4] = None

        # h5py.h5z.FILTER_SCALEOFFSET == 6
        self._hdf5_regfilters_subset[6] = None

        # LZO
        self._hdf5_regfilters_subset[305] = None

        # BZIP2
        self._hdf5_regfilters_subset[307] = numcodecs.BZ2

        # LZF
        self._hdf5_regfilters_subset[32000] = None

        # Blosc
        self._hdf5_regfilters_subset[32001] = numcodecs.Blosc

        # Snappy
        self._hdf5_regfilters_subset[32003] = None

        # LZ4
        self._hdf5_regfilters_subset[32004] = numcodecs.LZ4

        # bitshuffle
        self._hdf5_regfilters_subset[32008] = None

        # JPEG-LS
        self._hdf5_regfilters_subset[32012] = None

        # Zfp
        self._hdf5_regfilters_subset[32013] = None

        # Fpzip
        self._hdf5_regfilters_subset[32014] = None

        # Zstandard
        self._hdf5_regfilters_subset[32015] = numcodecs.Zstd

        # FCIDECOMP
        self._hdf5_regfilters_subset[32018] = None

    def copy_attrs_data_to_zarr_store(self, h5obj, zobj):
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
                    deref_obj = h5obj.file[val]
                    if deref_obj.name:
                        val = '//'+h5obj.file[val].name
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

    def storage_info(self, dset, dset_chunks):
        if dset.shape is None:
            # Null dataset
            return dict()

        dsid = dset.id
        if dset.chunks is None:
            # get offset for Non-External datasets
            if dsid.get_offset() is None:
                return dict()
            else:
                if dset_chunks is None:
                    key = (0,) * (len(dset.shape) or 1)
                    return {key: {'offset': dsid.get_offset(),
                                  'size': dsid.get_storage_size()}}
                else:
                    stinfo = dict()

                    bytes_offset = dsid.get_offset()
                    storage_size = dsid.get_storage_size()
                    key = (0,)*len(dset_chunks)

                    offsets_, sizes_, chunk_indices = self._get_chunkstorage_info(dset, bytes_offset, dset.shape,
                                                                                  storage_size, dset_chunks, key)

                    for i in range(len(chunk_indices)):
                        stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                         'size': sizes_[i]}

                    return stinfo

        else:
            # Currently, this function only gets the number of all written chunks, regardless of the dataspace.
            # HDF5 1.10.5
            # TO DO #
            num_chunks = dsid.get_num_chunks()

            if num_chunks == 0:
                return dict()

            stinfo = dict()
            chunk_size = dset.chunks
            if dsid.get_space_status() == h5py.h5d.SPACE_STATUS_ALLOCATED:
                chunk_maxind = np.ceil([a / b for a, b in zip(dset.shape, dset_chunks)]).astype(int)
                chunk_indices = np.indices(chunk_maxind)\
                                  .transpose(*range(1, len(chunk_maxind)+1), 0)\
                                  .reshape(np.prod(chunk_maxind), len(chunk_maxind))
                chunk_indices = [tuple(c) for c in chunk_indices]

                _get_chunk_info_by_coord = dsid.get_chunk_info_by_coord
                if dset_chunks == chunk_size:
                    for key in chunk_indices:
                        blob = _get_chunk_info_by_coord(tuple(np.array(key)*chunk_size))

                        stinfo[key] = {'offset': blob.byte_offset,
                                       'size': blob.size}
                else:
                    for key in chunk_indices:
                        blob = _get_chunk_info_by_coord(tuple(np.array(key)*dset_chunks))

                        bytes_offset = blob.byte_offset
                        blob_size = blob.size

                        offsets_, sizes_, chunk_indices = self._get_chunkstorage_info(dset, bytes_offset, chunk_size,
                                                                                      blob_size, dset_chunks, key)

                        for i in range(len(chunk_indices)):
                            stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                             'size': sizes_[i]}

            else:
                _get_chunk_info = dsid.get_chunk_info
                if dset_chunks == chunk_size:
                    for index in range(num_chunks):
                        blob = _get_chunk_info(index)

                        key = tuple([a // b for a, b in zip(blob.chunk_offset, chunk_size)])

                        stinfo[key] = {'offset': blob.byte_offset,
                                       'size': blob.size}
                else:
                    for index in range(num_chunks):
                        blob = _get_chunk_info(index)
                        key = tuple([a // b for a, b in zip(blob.chunk_offset, chunk_size)])

                        bytes_offset = blob.byte_offset
                        blob_size = blob.size

                        offsets_, sizes_, chunk_indices = self._get_chunkstorage_info(dset, bytes_offset, chunk_size,
                                                                                      blob_size, dset_chunks, key)

                        for i in range(len(chunk_indices)):
                            stinfo[(*chunk_indices[i], )] = {'offset': offsets_[i],
                                                             'size': sizes_[i]}

            return stinfo

    def _get_chunkstorage_info(self, dset, bytes_offset, blob_shape, blob_size, dset_chunks, key):

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
            # TO DO #

            for key in info:
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

        if (not isinstance(h5py_group, h5py.File) and (not isinstance(h5py_group, h5py.Group) or
           not isinstance(self.file.get(h5py_group.name, getlink=True), h5py.HardLink))):
            raise TypeError(f"{h5py_group} should be a h5py.File or h5py.Group as a h5py.HardLink")

        if h5py.version.hdf5_version_tuple < (1, 10, 5):
            raise Exception(("HDF5Zarr requires h5py installed with minimum hdf5 version of 1.10.5,\n"
                             f"Current hdf5 version {h5py.version.hdf5_version},\n"
                             "h5py installation: https://h5py.readthedocs.io/en/stable/build.html#custom-installation"))

        def _get_address(name, info):
            obj = self.group[name]
            self._address_dict[info.addr] = obj.name

        # add object addresses in file to self._address_dict
        self._address_dict[h5py.h5o.get_info(h5py_group.id).addr] = h5py_group.name
        h5py.h5o.visit(self.file.id, _get_address, obj_name=bytes(self.group.name, encoding='utf-8'), info=True)
        fcid = self.file.id.get_create_plist()
        unit_address_size, unit_length_size = fcid.get_sizes()
        self._address_dict['source'] = {'uri': self.uri,
                                        'offset_byte_size': unit_address_size,
                                        'length_byte_size': unit_length_size,
                                        'userblock_size': self.file.userblock_size}

        FileChunkStore.obj_address_info(self.store, self._address_dict)

        def _visit_create_zarr_hierarchy(name, link_info):
            obj = self.group[name]

            # Datasets
            if isinstance(obj, h5py.Dataset):
                # TO DO, Soft Links #
                if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                    print(f"Dataset {obj.name} is not processed: External Link")
                    return None

                self._create_zarr_hierarchy(obj, self.zgroup)
            # Groups
            elif isinstance(obj, h5py.Group):
                if link_info.type == h5py.h5l.TYPE_EXTERNAL:
                    print(f"Group {obj.name} is not processed: External Link")
                    return None

                zgroup_ = self.zgroup.create_group(name, overwrite=True)
                if link_info.type == h5py.h5l.TYPE_SOFT:
                    zgroup_path = zgroup_.create_group(SYMLINK, overwrite=True)
                    zgroup_path.attrs[obj.name] = self.file.get(obj.name, getlink=True).path

            self.copy_attrs_data_to_zarr_store(obj, self.zgroup[name])

        # create zarr hierarchy
        self.copy_attrs_data_to_zarr_store(self.group, self.zgroup)
        self.file.id.links.visit(_visit_create_zarr_hierarchy, obj_name=bytes(self.group.name, encoding='utf-8'), info=True)

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
                        dset_fillvalue[dt_i] = h5py.h5o.get_info([h5py.h5r.dereference(
                                                                  dset.fillvalue[dt_i], self.file.id)]).addr
                    else:
                        dset_fillvalue[dt_i] = 0
                else:
                    dtype_ += [(dtname, dset.dtype.base[dt_i])]

            dtype_ = np.dtype(dtype_)
            # currently not using the fill value with structured array
            # containing object, zarr v2.4.1, zarr PR 422
            dset_fillvalue = None

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

        info = self.storage_info(dset, dset_chunks)

        # Store metadata
        if info:
            info['source'] = {'uri': self.uri,
                              'array_name': dset.name}

            if dtype_refs is not None:
                info['source'].update(dtype_refs)
            if object_codec is not None:
                info.update(self.vlen_storage_info(dset, info))

            FileChunkStore.chunks_info(zarray, info)

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
        obj_path: str
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

    def _ensure_dict(self, obj):
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
            chunk_loc = zchunks[chunk_key]
        except KeyError:
            raise KeyError(chunk_key)

        # Read chunk's data...

        self._source.seek(chunk_loc['offset'], os.SEEK_SET)
        bytes = self._source.read(chunk_loc['size'])

        # variable-length string
        if 'gcol_offsets' in chunk_loc:

            data_array = np.frombuffer(bytes, dtype=self.dt_vlen)
            data_offsets = np.unique(data_array['address'])
            data_offsets.sort()
            data_offsets = data_offsets.tolist()

            data_bytes = np.empty(shape=len(data_offsets)+1, dtype=object)
            data_bytes[0] = bytes

            for i in range(len(data_offsets)):
                offset_item = data_offsets[i]
                if offset_item not in self._gcol:
                    gcol_size, skip = chunk_loc['gcol_offsets'][str(offset_item)]
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
            address_key = self._get_reference_key('')
            address_key = self._ensure_dict(self._store[address_key])
            address_key_source = address_key.pop('source')
            address_key = {int(k): v for k, v in address_key.items()}

            # function to convert references to string
            ref_array_func = np.frompyfunc(lambda x: address_key[int(x)], 1, 1)
            if dtype_str == '|O':

                if zref_dtype != "Object Reference":
                    raise TypeError
                dtype_ = np.dtype(f'uint{address_key_source["offset_byte_size"]*8}')
                data_array = np.frombuffer(bytes, dtype=dtype_)
                data_bytes = np.empty(shape=zarray_key['shape'], dtype=dtype_str)
                data_bytes[...] = ref_array_func(data_array)
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

                data_array = np.frombuffer(bytes, dtype=dtype_)
                data_bytes = np.empty(shape=zarray_key['shape'], dtype=dtype_str)

                for dtname, dt in zref_dtype.items():
                    if dt == "Object Reference":
                        data_bytes[dtname] = ref_array_func(data_array[dtname])
                    else:
                        data_bytes[dtname] = data_array[dtname]

            return data_bytes
        else:
            return bytes

    def _get_array_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / array_meta_key)

    def _get_reference_key(self, chunk_key):
        return str(PurePosixPath(chunk_key).parent / reference_key)

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
