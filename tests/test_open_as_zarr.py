from numpy.testing import assert_array_equal
import numpy as np
import h5py
from hdf5zarr import open_as_zarr
import pytest
import fsspec
import zarr


class Test_open_as_zarr_dset(object):
    """ independently test open_as_zarr with h5py.Dataset as filename argument """

    def test_open_as_zarr_dset_values(self, request, capsys, filesbase):
        # get list of files passes by hdf5files arg
        # if hdf5files is not specified, file_list will contain generated hdf5 files
        num_files = filesbase.num_files
        file_list = filesbase.file_list[0:num_files]

        # find list of datasets in files
        if len(self.objnames) != 0:
            dset_list = [f[name] for name in self.objnames for f in file_list if (name in f and isinstance(f[name], h5py.Dataset))]
            if len(dset_list) == 0:
                message = f"No given file contains {self.objnames}"
                with capsys.disabled():
                    print("\n"+message.rjust(len(request.node.nodeid)), end='')
                pytest.skip(message)
        # if objnames arg is not passed, select datasets for each file
        else:
            numdset = request.config.getoption('numdataset')
            if numdset <= 0:
                pytest.skip(f"numdataset: {numdset}")

            dset_names = []

            def _get_dsets(name, info):
                nonlocal dset_names
                if info.type == h5py.h5o.TYPE_DATASET:
                    dset_names.append(name.decode('utf-8'))

            dset_list = []
            for hfile in file_list:
                dset_names = []
                h5py.h5o.visit(hfile.id, _get_dsets, info=True)
                names = dset_names[0:numdset]
                for name in names:
                    dset_list.append(hfile[name])
                    message = f"objnames not specified. open_as_zarr run with file: {hfile.filename}, dataset: {name}"
                    with capsys.disabled():
                        print("\n"+message.rjust(len(request.node.nodeid)), end='')

        for dset in dset_list:
            with capsys.disabled():
                print("\n"+f"file: {dset.file.filename}, dataset: {dset}  :".rjust(len(request.node.nodeid)), end='')
                print("\n"+f"dataset: {dset.name}, data  :".rjust(len(request.node.nodeid)), end='')

            # call open_as_zarr
            if not dset.dtype.hasobject:
                zarray = open_as_zarr(dset)  # dataset does not have object references
            else:
                zarray = open_as_zarr(dset, collectrefs=True)  # dataset with object references

            # test values when dtype is variable length
            if h5py.check_vlen_dtype(dset.dtype) is not None:
                dset_str = dset.asstr()  # wrapper to read data as python str
                assert_array_equal(dset_str, zarray)
            # test values when dtype is structured array with object reference
            elif dset.dtype.hasobject and dset.dtype.names is not None:
                hval = dset[()]
                # function to get reference target names
                ref_array_func = np.frompyfunc(lambda x: h5py.h5i.get_name(h5py.h5r.dereference(x, dset.file.id)), 1, 1)
                for dtname in dset.dtype.names:
                    if dset.dtype[dtname].hasobject:
                        if dset.shape != ():
                            hval_str = ref_array_func(hval[dtname]).astype(str)
                        else:
                            hval_str = h5py.h5i.get_name(h5py.h5r.dereference(hval[dtname], dset.file.id))
                            hval_str = hval_str.decode('utf-8')
                        assert_array_equal(hval_str, zarray[dtname])
            # test values when dtype is object reference
            elif dset.dtype.hasobject and dset.dtype.names is None:
                hval = dset[()]
                # function to get reference target names
                ref_array_func = np.frompyfunc(lambda x: h5py.h5i.get_name(h5py.h5r.dereference(x, dset.file.id)), 1, 1)
                if dset.shape != ():
                    hval_str = ref_array_func(hval).astype(str)
                else:
                    hval_str = h5py.h5i.get_name(h5py.h5r.dereference(hval, dset.file.id))
                    hval_str = hval_str.decode('utf-8')
                assert_array_equal(hval_str, zarray)
            # test values when dtype is simple
            else:
                assert_array_equal(dset, zarray)

            with capsys.disabled():
                print("\n"+f"dataset: {dset.name}, attrs  :".rjust(len(request.node.nodeid)), end='')

            # test attrs
            for key, val in dset.attrs.items():
                assert_array_equal(val, zarray.attrs[key])

            with capsys.disabled():
                print("\n"+f"dataset: {dset.name}, fillvalue  :".rjust(len(request.node.nodeid)), end='')

            # test fillvalue
            # if dtype is structured array
            if dset.fillvalue is not None and dset.fillvalue.dtype.names is not None:
                if dset.fillvalue.dtype.hasobject:
                    message = f"structured array fillvalue {dset.fillvalue} with object dtype not supported."
                    with capsys.disabled():
                        print(("\n"+message).rjust(len(request.node.nodeid)), end='')
                    pytest.xfail(reason=message)
                assert_array_equal(dset.fill_value, zarray.fillvalue)
            # if fillvalue is an object reference:
            elif dset.fillvalue is not None and dset.fillvalue.dtype.hasobject:
                hval_str = h5py.h5i.get_name(h5py.h5r.dereference(dset.fillvalue, dset.file.id))
                hval_str = hval_str.decode('utf-8')
                assert_array_equal(hval_str, zarray.fill_value)
            # simple fillvalue
            else:
                assert_array_equal(dset.fillvalue, zarray.fill_value)

    def test_open_as_zarr_remote(self, request, capsys):
        # remote test with ros3 and open_as_zarr
        item = 'https://dandiarchive.s3.amazonaws.com/girder-assetstore/4f/5a/4f5a24f7608041e495c85329dba318b7'
        dsetname = '/acquisition/raw_running_wheel_rotation/data'
        if 'ros3' in h5py.registered_drivers():
            hfile = h5py.File(item, mode='r', driver='ros3')
        else:
            f = fsspec.open(item, 'rb')
            hfile = h5py.File(f.open(), mode='r')

        dset = hfile[dsetname]

        with capsys.disabled():
            print("\n"+f"{item}  :".rjust(len(request.node.nodeid)), end='')
            print("\n"+f"dataset: {dsetname}, data  :".rjust(len(request.node.nodeid)), end='')

        zarray = open_as_zarr(dset)  # dataset does not have object references
        assert isinstance(zarray, zarr.Array)

        # test simple dtype
        assert_array_equal(dset, zarray)

        with capsys.disabled():
            print("\n"+f"dataset: {dset.name}, fillvalue  :".rjust(len(request.node.nodeid)), end='')

        # test simple fillvalue
        assert_array_equal(dset.fillvalue, zarray.fill_value)
