from numpy.testing import assert_array_equal
import numpy as np
import tempfile
import secrets
import h5py
from hdf5zarr import HDF5Zarr
import pytest
import itertools


class HDF5ZarrBase(object):

    ##########################################
    #  basic tests                           #
    ##########################################

    @pytest.mark.parametrize('visit_type', [None], ids=[""])
    def test_consolidate_metadata(self):
        zgroup = self.hdf5zarr.consolidate_metadata()

    def test_groups(self):
        """ test if group exists """
        def _test_groups(name, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_GROUP:
                self.zgroup[name]
                self.zgroup[name.decode('utf-8')]
        h5py.h5o.visit(self.hfile.id, _test_groups, info=True)

    def test_dsets(self):
        """ test if dataset exists """
        def _test_dsets(name, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_DATASET:
                self.zgroup[name]
                self.zgroup[name.decode('utf-8')]
        h5py.h5o.visit(self.hfile.id, _test_dsets, info=True)

    ##########################################
    #  dataset properties tests              #
    ##########################################

    def test_dset_properties_dtype(self):
        """ test if dataset dtypes are equal """
        def _test_dtype(zobj, hobj, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_DATASET:
                assert zobj.dtype == hobj.dtype
        self._visit_item(_test_dtype)

    def test_dset_properties_shape(self):
        """ test if dataset shapes are equal """
        def _test_shape(zobj, hobj, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_DATASET:
                assert zobj.shape == hobj.shape
        self._visit_item(_test_shape)

    def test_dset_properties_chunks(self):
        """ test if datasets properties are equal """
        def _test_chunks(zobj, hobj, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_DATASET:
                if hobj.chunks is None:
                    chunks = tuple(s if s != 0 else 1 for s in hobj.shape)
                else:
                    chunks = hobj.chunks
                assert zobj.chunks == chunks
        # skip if max_chunksize is not None
        if self.hdf5zarr.max_chunksize is None:
            self._visit_item(_test_chunks)

    def test_dset_properties_fillvalue(self):
        """ test if datasets properties are equal """
        def _test_fillvalue(zobj, hobj, hobj_info):
            if hobj_info.type == h5py.h5o.TYPE_DATASET:
                # skip test for structured array containing object ref, zarr v2.4.1, zarr PR 422
                dset_type = hobj.id.get_type()
                if hobj.dtype.names is None:
                    assert_array_equal(zobj.fill_value, hobj.fillvalue)
                elif all([dset_type.get_member_class(i) != h5py.h5t.REFERENCE for i in range(dset_type.get_nmembers())]):
                    for n in self.hobj.dtype.names:
                        assert_array_equal(zobj.fill_value[n], hobj.fillvalue[n])
        self._visit_item(_test_fillvalue)

    ##########################################
    #  dataset read tests                    #
    ##########################################

    def test_zarray_read_simple_dtype(self):
        """ test if zarr arrays are read """
        def _test_dsets_read(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, False) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                zval = zobj[()]
        self._visit_item(_test_dsets_read)

    def test_zarray_read_simple_dtype_objref(self):
        """ test if zarr struct arrays are read """
        def _test_dsets_read(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, True) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                zval = zobj[()]
        self._visit_item(_test_dsets_read)

    def test_zarray_read_struct_dtype_noref(self):
        """ test if zarr struct arrays are read """
        def _test_dsets_read(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (True, False) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                zval = zobj[()]
        self._visit_item(_test_dsets_read)

    def test_zarray_read_struct_dtype_withobjref(self):
        """ test if zarr struct arrays are read """
        def _test_dsets_read(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (True, True) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                zval = zobj[()]
        self._visit_item(_test_dsets_read)

    def test_zarray_read_vlenstring(self):
        """ test if variable length string arrays are read """
        def _test_dsets_read(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, False) and
               h5py.check_vlen_dtype(hobj.dtype)):
                zval = zobj[()]
        self._visit_item(_test_dsets_read)

    def test_dset_val_simple_dtype(self):
        """ test if zarr arrays and datasets are equal """
        def _test_dset_val(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, False) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                hval = hobj[()]
                zval = zobj[()]
                assert_array_equal(hval, zval)
        self._visit_item(_test_dset_val)

    def test_dset_val_simple_dtype_objref(self):
        """ test if zarr arrays and datasets are equal """
        def _test_dset_val(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, True) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                hval = hobj[()]
                zval = zobj[()]
                ref_array_func = np.frompyfunc(lambda x: h5py.h5i.get_name(h5py.h5r.dereference(x, self.hfile.id)), 1, 1)
                if hobj.shape != ():
                    hval_str = ref_array_func(hval).astype(str)
                else:
                    hval_str = h5py.h5i.get_name(h5py.h5r.dereference(hval, self.hfile.id))
                    hval_str = hval_str.decode('utf-8')
                if self.hfile.name == '/':
                    assert_array_equal(hval_str, zval)
                else:
                    assert_array_equal(np.frompyfunc(lambda x: x if x.startswith(self.hfile.name) else '', 1, 1)(hval_str), zval)
        self._visit_item(_test_dset_val)

    def test_dset_val_struct_dtype_noref(self):
        """ test if zarr arrays and datasets are equal """
        def _test_dset_val(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (True, False) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                hval = hobj[()]
                zval = zobj[()]
                for dt_name in hobj.dtype.names:
                    assert_array_equal(hval[dt_name], zval[dt_name])
        self._visit_item(_test_dset_val)

    def test_dset_val_struct_dtype_withobjref(self):
        """ test if zarr arrays and datasets are equal """
        def _test_dset_val(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (True, True) and
               not h5py.check_vlen_dtype(hobj.dtype)):
                hval = hobj[()]
                zval = zobj[()]
                ref_array_func = np.frompyfunc(lambda x: h5py.h5i.get_name(h5py.h5r.dereference(x, self.hfile.id)), 1, 1)
                dset_type = hobj.id.get_type()
                for i in range(dset_type.get_nmembers()):
                    dt_name = hobj.dtype.names[i]
                    if dset_type.get_member_class(i) == h5py.h5t.REFERENCE:
                        if hobj.shape != ():
                            hval_str = ref_array_func(hval[dt_name]).astype(str)
                        else:
                            hval_str = h5py.h5i.get_name(h5py.h5r.dereference(hval[dt_name], self.hfile.id))
                            hval_str = hval_str.decode('utf-8')
                        if self.hfile.name == '/':
                            assert_array_equal(hval_str, zval[dt_name])
                        else:
                            assert_array_equal(np.frompyfunc(lambda x: x if x.startswith(self.hfile.name)
                                                                         else '', 1, 1)(hval_str), zval[dt_name])
                    else:
                        assert_array_equal(hval[dt_name], zval[dt_name])
        self._visit_item(_test_dset_val)

    def test_dset_val_vlenstring(self):
        """ test if zarr arrays and datasets are equal """
        def _test_dset_val(zobj, hobj, hobj_info):
            if (hobj_info.type == h5py.h5o.TYPE_DATASET and
               self._checkdtype_structobjref(hobj) == (False, False) and
               h5py.check_vlen_dtype(hobj.dtype)):
                hobj = hobj.asstr()  # wrapper to read data as python str
                hval = hobj[()]
                zval = zobj[()]
                assert_array_equal(hval, zval)
        self._visit_item(_test_dset_val)

    def test_attrs(self):
        """ test if attributes exist """
        def _test_attr(zobj, hobj, hobj_info):
            for name in hobj.attrs:
                zattr = zobj.attrs[name]
        self._visit_item(_test_attr)

    def test_read_attrs(self):
        """ test if attributes are equal """
        def _test_read_attrs(zobj, hobj, hobj_info):
            for name in hobj.attrs:
                hattr = hobj.attrs[name]
                zattr = zobj.attrs[name]
                assert_array_equal(zattr, hattr)
        self._visit_item(_test_read_attrs)

    @pytest.fixture(autouse=True)
    def visit_files(self, request):
        # file number
        fnum = request.node.callspec.params['fnum']
        self.hfile, self.hdf5zarr = self.file_list[fnum], self.hdf5zarr_list[fnum]
        self.zgroup = self.hdf5zarr.zgroup

    # visit hdf5 items
    # visit types, objects or links
    @pytest.fixture(autouse=True, params=["objects_only", "links_only"])
    def visit_type(self, request):
        self.visittype = request.param

        # collect flag
        self.fkeep = request.config.getoption("fkeep")

        self._ex = []

        # visit objects
        if self.visittype == "objects_only":
            self._visit_item = self.visit_obj_func
        elif self.visittype == "links_only":
            self._visit_item = self.visit_link_func
        elif self.visittype is None:
            pass
        else:
            raise Exception("Invalid visit_type parameter")

        yield

        if request.node.rep_setup.passed:
            if request.node.rep_call.failed:
                if self.fkeep and len(self._ex) > 0:
                    ex, name = self._ex[0]
                    hobj = self.hfile[name]
                else:
                    hobj = self.hobj
                print(f"""HDF5Zarr args: (
                       filename = '{self.hdf5zarr.filename}',
                       store = {self.hdf5zarr.store},
                       store_mode = {self.hdf5zarr.store_mode},
                       max_chunksize = {self.hdf5zarr.max_chunksize},
                       )""")

                print("executing test failed for", request.node.name, hobj.file.filename)
                if hobj.file.filename != self._testfilename and isinstance(hobj, h5py.Dataset):
                    print(f"""hdf5 Dataset: (
                           name = '{hobj.name}',
                           shape = {hobj.shape},
                           dtype = {hobj.dtype},
                           chunks = {hobj.chunks},
                           maxshape = {hobj.maxshape},
                           track_times = None,
                           track_order = None,
                           fillvalue = {hobj.fillvalue},
                           data = {hobj[()]},
                           )""")

                if self.fkeep and len(self._ex) > 1:
                    ex_m, name_m = self._ex[np.argmin([self.hfile[name].size for ex, name in self._ex])]
                    if name_m != name:
                        hobj = self.hfile[name_m]
                        print("executing test failed for", request.node.name, hobj.file.filename)
                        if hobj.file.filename != self._testfilename and isinstance(hobj, h5py.Dataset):
                            print(f"""(
                                   name = '{hobj.name}',
                                   shape = {hobj.shape},
                                   dtype = {hobj.dtype},
                                   chunks = {hobj.chunks},
                                   maxshape = {hobj.maxshape},
                                   track_times = None,
                                   track_order = None,
                                   fillvalue = {hobj.fillvalue},
                                   data = {hobj[()]},
                                   )""")

    def visit_obj_func(self, assert_func):
        # visit objects

        _ex = []

        def _test_obj(name, hobj_info):
            nonlocal _ex

            self.hobj = self.hfile[name]
            self.zobj = self.zgroup[name.decode('utf-8')]
            if not self.fkeep:
                assert_func(self.zobj, self.hobj, hobj_info)
            else:
                try:
                    assert_func(self.zobj, self.hobj, hobj_info)
                except AssertionError as ex:
                    _ex.append([ex, name])

        if self.objnames == []:
            h5py.h5o.visit(self.hfile.id, _test_obj, info=True)
        else:
            for name in self.objnames:
                hobj_info = h5py.h5g.get_objinfo(self.hfile[name].id)
                _test_obj(name, hobj_info)

        self._ex = _ex

        # raise only one exception in case of fkeep == True
        if self.fkeep and len(self._ex) > 0:
            raise self._ex[0][0]

    def visit_link_func(self, assert_func):
        # visit links

        _ex = []

        def _test_obj(name, hlink_info):
            nonlocal _ex

            if not isinstance(self.hfile, h5py.Dataset):
                self.hobj = self.hfile[name]
                self.zobj = self.zgroup[name.decode('utf-8')]
            else:
                self.hobj = self.hfile
                self.zobj = self.zgroup

            hobj_info = h5py.h5g.get_objinfo(self.hobj.id)

            if hlink_info.type == h5py.h5l.TYPE_SOFT:
                if not self.fkeep:
                    assert_func(self.zobj, self.hobj, hobj_info)
                else:
                    try:
                        assert_func(self.zobj, self.hobj, hobj_info)
                    except AssertionError as ex:
                        _ex.append([ex, name])

            else:
                # TO DO
                pass

        if self.objnames == []:
            if not isinstance(self.hfile, h5py.Dataset):
                self.hfile.id.links.visit(_test_obj, info=True)
            else:
                name = self.hfile.name
                hlink_info = self.hfile.file.id.links.get_info(bytes(name, encoding='utf-8'))
                _test_obj(name, hlink_info)
        else:
            for name in self.objnames:
                hlink_info = self.hfile.id.links.get_info(name)
                _test_obj(name, hlink_info)

        self._ex = _ex

        # raise only one exception in case of fkeep == True
        if self.fkeep and len(self._ex) > 0:
            raise self._ex[0][0]

    @classmethod
    def _checkdtype_structobjref(cls, hobj):
        """ return 2-tuple indicating struct array and object reference dtype """
        dset_type = hobj.id.get_type()
        if hobj.dtype.names is None:
            return (False,  # simple dtype
                    dset_type.get_class() == h5py.h5t.REFERENCE)
        else:
            return (True,  # struct dtype
                    any([dset_type.get_member_class(i) == h5py.h5t.REFERENCE for i in range(dset_type.get_nmembers())]))


class TestHDF5Zarr(HDF5ZarrBase):
    """ Comparing HDF5Zarr read with h5py """

    @classmethod
    def setup_class(cls):
        # list of numpy dtypes up to 8 bytes
        cls.attribute_dtypes = list(set(np.typeDict.values()) -
                                    set([np.void, np.str_, np.bytes_, np.object_, np.timedelta64,
                                         np.complex64, np.complex256, np.float128, np.complex128,
                                         np.datetime64]))
        cls.dset_dtypes = cls.attribute_dtypes
        cls.depth = 3  # nested groups depth

        # all n_* are per group or per object
        cls.n_dsets = 4  # number of regular (or scalar) datasets without object references or struct array dtypes in each group
        cls.n_groups = 3  # number of groups in each group
        cls.n_groupsoftlink = 1  # number of soft links to another group in each group
        cls.n_grouphardlink = 1  # TO DO number of hard links to another group in each group
        cls.n_dsetsoftlink = 1  # number of soft links to another dataset in each group
        cls.n_dsethardlink = 1  # TO DO number of hard links to another dataset in each group

        cls.n_objectrefdset = 1  # number of object reference datasets in each group
        cls.n_objectrefdsetmaxdim = 4  # maximum number of dimensions in an object reference datasets

        cls.n_structarraywithobjrefdset = 1  # number of struct array datasets containing object ref dtype in each group
        cls.n_structarrayobjrefdtype = 1  # number of object ref dtypes if used in a struct array

        cls.n_structarrayregulardset = 1  # number of struct array datasets without object refernce dtype in each group
        cls.n_structarraydtypelen = 4  # length of struct array dtypes in datasets

        cls.n_vlenstringdset = 1  # number of variable length string datasets in each group
        cls.n_vlenstringdsetmaxlen = 2000  # max length of strings in variable length datasets

        cls.n_dsetmaxdim = 4  # maximum number of dimensions

        cls.n_attributes_min = 5  # min number of attributes for each object

        cls.n_nulldsets_infile = 1  # TO DO number of null datasets in file

        cls.srand = secrets.SystemRandom()

        if cls.hdf5files_option:
            cls.file_list = [h5py.File(i, 'r') for i in cls.hdf5file_names]
        else:
            cls.file_list = [cls._create_file(i) for i in cls.hdf5file_names]

        cls.hdf5zarr_list = [HDF5Zarr(f.filename, max_chunksize=None) for f in cls.file_list]

        # prepend _testfile if hdf5files are not specified
        if not cls.hdf5files_option:
            cls.file_list.insert(0, cls._testfile())
            cls.hdf5zarr_list.insert(0, HDF5Zarr(cls.file_list[0].filename, max_chunksize=None))

        # track which temporary files are already saved.
        # if hdf5files_option is passed, mark them as already saved
        num_files = len(cls.file_list)
        cls.num_files = num_files
        cls.fnum_keep = {i: cls.hdf5files_option for i in range(0, num_files)}
        # do not save "_testfile"
        cls.fnum_keep[0] = True

        group_names = []
        dset_names = []
        def _get_objs(name, info):
            nonlocal group_names, dset_names
            if info.type == h5py.h5o.TYPE_GROUP:
                group_names.append(name.decode('utf-8'))
            elif info.type == h5py.h5o.TYPE_DATASET:
                dset_names.append(name.decode('utf-8'))

        if cls.numsubgroup != 0:
            # len(cls.ids_subgroup) == num_files*cls.numsubgroup
            cls.file_list += cls.file_list[:num_files]*cls.numsubgroup
            cls.hdf5zarr_list += [None]*num_files*cls.numsubgroup
            for i in range(num_files, num_files*(1+cls.numsubgroup)):
                group_names = []
                dset_names = []
                h5py.h5o.visit(cls.file_list[i].id, _get_objs, info=True)
                group_names.sort()
                dset_names.sort()
                obj_names = [_objname for _objname in itertools.chain(*itertools.zip_longest(group_names, dset_names))
                             if _objname is not None]  # interleave groups and dsets
                if len(obj_names) != 0:
                    # select next group in sorted group names
                    hdf5group = obj_names[(i-num_files)//num_files if len(obj_names) > (i-num_files)//num_files else -1]
                else:
                    hdf5group = None
                cls.hdf5zarr_list[i] = HDF5Zarr(cls.file_list[i].filename, hdf5group=hdf5group)
                cls.file_list[i] = cls.file_list[i][hdf5group or '/']

        if not cls.disable_max_chunksize:
            # len(cls.ids_maxchunksize) == num_files*int(not disable_max_chunksize)*cls.num_maxchunksize
            cls.file_list += cls.file_list[:num_files]*2
            cls.hdf5zarr_list += [None]*num_files*2
            for i in range(len(cls.file_list)-2*num_files, len(cls.file_list)-num_files):
                max_chunksize = 1000 if not cls.hdf5files_option else 2**cls.srand.randint(14, 20)
                cls.hdf5zarr_list[i] = HDF5Zarr(cls.file_list[i].filename, max_chunksize=max_chunksize)
            for i in range(len(cls.file_list)-num_files, len(cls.file_list)):
                max_chunksize = 2**cls.srand.randint(10, 20) if not cls.hdf5files_option else 2**cls.srand.randint(14, 20)
                cls.hdf5zarr_list[i] = HDF5Zarr(cls.file_list[i].filename, max_chunksize=max_chunksize)

    @classmethod
    def teardown_class(cls):
        for f in cls.file_list[:cls.num_files]:
            f.file.close()

    @classmethod
    def _create_file(cls, name):
        """ create test hdf5 file """

        srand = cls.srand

        # create hdf5 file
        cls.temp_file = tempfile.NamedTemporaryFile(suffix=".hdf5", prefix=name, delete=False)
        cls.temp_file.close()
        hfile = h5py.File(cls.temp_file.name, 'w')

        # create nested groups
        groupnames_prefix = [chr(65+i)for i in range(cls.n_groups)]  # e.g. ['A', 'B', 'C']
        group_list = [hfile]  # list containing all groups

        def _create_groups(obj, d):
            nonlocal group_list

            for c in groupnames_prefix:
                g_name = c + str(cls.depth - d)
                g = obj.create_group(g_name)
                group_list.append(g)
                if d > 0:
                    _create_groups(obj[g_name], d-1)

        _create_groups(hfile, cls.depth)

        # create softlinks to groups
        for g in group_list:
            for i in range(cls.n_groupsoftlink):
                # do not use rand_rng.choice
                target_str = srand.choice(group_list).name
                g[f"SoftLg{i}"] = h5py.SoftLink(target_str)

        # create datasets
        # TO DO, external dsets
        # TO DO, compression
        srand.shuffle(cls.dset_dtypes)
        iter_dtypes = itertools.cycle(cls.dset_dtypes)  # shuffle dtypes to cycle over when creating dsets
        iter_chunks = itertools.cycle([True, None])  # True or False cycle for auto chunking
        iter_track_times = itertools.cycle([False, True])  # True or False cycle for track_times
        iter_track_order = itertools.cycle([False, False, True, True])  # True or False cycle for track_order
        iter_fillvalue = itertools.cycle([None, True, True, None])  # True or False cycle for track_order
        rand_rng = np.random.default_rng()
        dset_list = []
        for g in group_list:
            # TO DO, add test with datasets with zero in dimensions
            for i in range(cls.n_dsets):
                shape = srand.choices(range(1, 90//(i or 1)), k=i)  # dseti has i dimensions
                size = np.prod(shape)
                dtype = next(iter_dtypes)
                if dtype == np.bool_:
                    data = np.frombuffer(rand_rng.bytes(size*8), dtype=np.int64) > 0
                elif dtype == np.datetime64:
                    data = np.datetime64('1970-01-01T00:00:00', 'ns') + np.frombuffer(rand_rng.bytes(size*8), dtype=np.uint64)
                    dtype = h5py.opaque_dtype(data.dtype)
                    data = data.astype(dtype)
                else:
                    data = np.frombuffer(rand_rng.bytes(size*np.dtype(dtype).itemsize), dtype=dtype)

                # create_dataset options comptability
                if len(shape) > 0:
                    chunks = next(iter_chunks)
                else:
                    chunks = None
                    # compression = None
                    # compression_opts = None
                    # shuffle = None
                    # fletcher32 = None
                    # scaleoffset = None
                fillvalue = None if (next(iter_fillvalue) is None or
                                     data.dtype.char == 'M') else data.reshape(size)[rand_rng.integers(0, size)]

                dset = g.create_dataset(
                           name='dset'+str(i),
                           shape=shape,
                           data=data,
                           dtype=dtype,
                           chunks=chunks,
                           maxshape=None if chunks is None else tuple(
                                          (np.array(shape) + rand_rng.integers(0, 5))*rand_rng.integers(1, 5, size=len(shape))),
                           track_times=next(iter_track_times),
                           track_order=next(iter_track_order),
                           fillvalue=fillvalue
                           )

                dset_list.append(dset)

        # create variable length string datasets
        for g in group_list:
            for i in range(cls.n_vlenstringdset):
                k = i + srand.randint(0, cls.n_dsetmaxdim)
                shape = srand.choices(range(1, int(60**(1/(k or 1)))+1), k=k)  # dseti has k dimensions
                size = int(np.prod(shape))
                dtype = h5py.string_dtype(encoding='utf-8')
                str_len = np.frombuffer(rand_rng.bytes(size*8), dtype=np.uint64) % cls.n_vlenstringdsetmaxlen
                data = np.array([''.join([chr(rand_rng.integers(0x0020, 0x03ff)) for _ in range(i)])
                                 for i in str_len], dtype=dtype)

                # create_dataset options comptability
                if len(shape) > 0:
                    chunks = next(iter_chunks)
                else:
                    chunks = None
                    # compression = None
                    # compression_opts = None
                    # shuffle = None
                    # fletcher32 = None
                    # scaleoffset = None
                fillvalue = None

                dset = g.create_dataset(
                           name='dsetvlenstring'+str(i),
                           shape=shape,
                           data=data,
                           dtype=dtype,
                           chunks=chunks,
                           maxshape=None if chunks is None else tuple(
                                          (np.array(shape) + rand_rng.integers(0, 5))*rand_rng.integers(1, 5, size=len(shape))),
                           track_times=next(iter_track_times),
                           track_order=next(iter_track_order),
                           fillvalue=fillvalue
                           )

                dset_list.append(dset)

        # create struct array datasets
        for g in group_list:
            # TO DO, add test with datasets with zero in dimensions
            for i in range(cls.n_structarrayregulardset):
                k = i + srand.randint(0, cls.n_dsetmaxdim)
                shape = srand.choices(range(1, 90//(k or 1)), k=k)  # dseti has k dimensions
                size = int(np.prod(shape))
                dtype = [(chr(97+j), next(iter_dtypes)) for j in range(cls.n_structarraydtypelen)]
                data = np.empty(shape=shape, dtype=dtype)
                for j in range(len(dtype)):
                    dt_name, dt = dtype[j]
                    if dt == np.bool_:
                        data_ = np.frombuffer(rand_rng.bytes(size*8), dtype=np.int64) > 0
                    elif dt == np.datetime64:
                        data_ = np.datetime64('1970-01-01T00:00:00', 'ns')+np.frombuffer(rand_rng.bytes(size*8), dtype=np.uint64)
                        dtype[j] = (dt_name, h5py.opaque_dtype(data_.dtype))
                        data_ = data_.astype(dtype[j][1])
                    else:
                        data_ = np.frombuffer(rand_rng.bytes(size*np.dtype(dt).itemsize), dtype=dt)

                    data[dt_name] = data_.reshape(shape)

                # create_dataset options comptability
                if len(shape) > 0:
                    chunks = next(iter_chunks)
                else:
                    chunks = None
                    # compression = None
                    # compression_opts = None
                    # shuffle = None
                    # fletcher32 = None
                    # scaleoffset = None
                fillvalue = None if (next(iter_fillvalue) is None or
                                     data.dtype.char == 'M') else data.reshape(size)[rand_rng.integers(0, size)]

                dset = g.create_dataset(
                           name='dsetstructarray'+str(i),
                           shape=shape,
                           data=data,
                           dtype=dtype,
                           chunks=chunks,
                           maxshape=None if chunks is None else tuple(
                                          (np.array(shape) + rand_rng.integers(0, 5))*rand_rng.integers(1, 5, size=len(shape))),
                           track_times=next(iter_track_times),
                           track_order=next(iter_track_order),
                           fillvalue=fillvalue
                           )

                dset_list.append(dset)

        # create object reference datasets
        for g in group_list:
            for i in range(cls.n_objectrefdset):
                k = i + srand.randint(0, cls.n_objectrefdsetmaxdim)
                shape = srand.choices(range(1, int(60**(1/(k or 1)))+1), k=k)  # dseti has k dimensions
                size = int(np.prod(shape))
                dtype = h5py.ref_dtype

                obj_list = dset_list + group_list
                data = np.array([srand.choice(obj_list).ref for _ in range(size)])

                # create_dataset options comptability
                if len(shape) > 0:
                    chunks = next(iter_chunks)
                else:
                    chunks = None
                    # compression = None
                    # compression_opts = None
                    # shuffle = None
                    # fletcher32 = None
                    # scaleoffset = None

                fillvalue = None

                dset = g.create_dataset(
                           name='dsetobjref'+str(i),
                           shape=shape,
                           data=data,
                           dtype=dtype,
                           chunks=chunks,
                           maxshape=None if chunks is None else tuple(
                                          (np.array(shape) + rand_rng.integers(0, 5))*rand_rng.integers(1, 5, size=len(shape))),
                           track_times=next(iter_track_times),
                           track_order=next(iter_track_order),
                           fillvalue=fillvalue
                           )

                dset_list.append(dset)

        # create struct array datasets with object reference
        for g in group_list:
            # TO DO, add test with datasets with zero in dimensions
            for i in range(cls.n_structarraywithobjrefdset):
                k = i + srand.randint(0, cls.n_objectrefdsetmaxdim)
                shape = srand.choices(range(1, int(60**(1/(k or 1)))+1), k=k)  # dseti has i dimensions
                size = int(np.prod(shape))
                dtypeobjind = rand_rng.choice(range(cls.n_structarraydtypelen), size=cls.n_structarrayobjrefdtype, replace=False)
                dtype = [(chr(97+j), h5py.ref_dtype if j in dtypeobjind else next(iter_dtypes))
                         for j in range(cls.n_structarraydtypelen)]

                data = np.empty(shape=shape, dtype=dtype)
                for j in range(len(dtype)):
                    dt_name, dt = dtype[j]
                    if dt == h5py.ref_dtype:
                        obj_list = dset_list + group_list
                        data_ = np.array([srand.choice(obj_list).ref for _ in range(size)])
                    elif dt == np.bool_:
                        data_ = np.frombuffer(rand_rng.bytes(size*8), dtype=np.int64) > 0
                    elif dt == np.datetime64:
                        data_ = np.datetime64('1970-01-01T00:00:00', 'ns')+np.frombuffer(rand_rng.bytes(size*8), dtype=np.uint64)
                        dtype[j] = (dt_name, h5py.opaque_dtype(data_.dtype))
                        data_ = data_.astype(dtype[j][1])
                    else:
                        data_ = np.frombuffer(rand_rng.bytes(size*np.dtype(dt).itemsize), dtype=dt)

                    data[dt_name] = data_.reshape(shape)

                # create_dataset options comptability
                if len(shape) > 0:
                    chunks = next(iter_chunks)
                else:
                    chunks = None
                    # compression = None
                    # compression_opts = None
                    # shuffle = None
                    # fletcher32 = None
                    # scaleoffset = None
                fillvalue = None if (next(iter_fillvalue) is None or
                                     data.dtype.char == 'M') else data.reshape(size)[rand_rng.integers(0, size)]

                dset = g.create_dataset(
                           name='dsetstructarraywobjref'+str(i),
                           shape=shape,
                           data=data,
                           dtype=dtype,
                           chunks=chunks,
                           maxshape=None if chunks is None else tuple(
                                          (np.array(shape) + rand_rng.integers(0, 5))*rand_rng.integers(1, 5, size=len(shape))),
                           track_times=next(iter_track_times),
                           track_order=next(iter_track_order),
                           fillvalue=fillvalue
                           )

                dset_list.append(dset)

        # create softlinks to datasets
        for g in group_list:
            for i in range(cls.n_dsetsoftlink):
                # do not use rand_rng.choice
                target_str = srand.choice(dset_list).name
                g[f"SoftLd{i}"] = h5py.SoftLink(target_str)

        # add attributes
        srand.shuffle(cls.dset_dtypes)
        iter_dtypes = itertools.cycle(cls.dset_dtypes)  # shuffle dtypes to cycle over when creating attributes
        for obj in itertools.chain(group_list, dset_list):
            for i in range(rand_rng.integers(cls.n_attributes_min, 26, endpoint=True)):
                dtype = next(iter_dtypes)
                attr_name = chr(97+i)
                if dtype == np.bool_:
                    attr = np.frombuffer(rand_rng.bytes(8), dtype=np.int64) > 0
                elif dtype == np.datetime64:
                    continue
                else:
                    attr = np.frombuffer(rand_rng.bytes(np.dtype(dtype).itemsize), dtype=dtype)
                obj.attrs[attr_name] = attr[0]

            # add array attributes
            for i in range(rand_rng.integers(cls.n_attributes_min, 26, endpoint=True)):
                shape = srand.choices(range(1, 10//(i//5 or 1)), k=i//5)  # attributes has i//5 dimensions
                size = np.prod(shape)
                dtype = next(iter_dtypes)
                attr_name = chr(65+i) + '_array_attr'
                if dtype == np.bool_:
                    attr = np.frombuffer(rand_rng.bytes(size*8), dtype=np.int64) > 0
                elif dtype == np.datetime64:
                    attr = np.datetime64('1970-01-01T00:00:00', 'ns') + np.frombuffer(rand_rng.bytes(size*8), dtype=np.uint64)
                    attr = attr.astype(h5py.opaque_dtype(attr.dtype))
                else:
                    attr = np.frombuffer(rand_rng.bytes(size*np.dtype(dtype).itemsize), dtype=dtype)
                obj.attrs[attr_name] = attr

        return hfile

    @classmethod
    def _testfile(cls):
        """ create test hdf5 file """

        # repeatable test
        # create hdf5 file
        cls.temp_file = tempfile.NamedTemporaryFile(suffix=".hdf5", prefix=cls._testfilename, delete=False)
        cls.temp_file.close()
        hfile = h5py.File(cls.temp_file.name, 'w')

        hfile.create_dataset(
            name='dset0',
            shape=(1,),
            data=[0],
            dtype=np.float,
            chunks=None,
            maxshape=None,
            track_times=None,
            track_order=None,
            fillvalue=1,
            )

        hfile.create_dataset(
            name='dsetscalar',
            shape=(),
            data=2,
            dtype=np.float,
            maxshape=None,
            track_times=None,
            track_order=None,
            fillvalue=1,
            )

        shape = (20, 3, 4, 13)
        data = np.array([cls.srand.choice([v for k, v in hfile.items()]).ref for _ in range(np.prod(shape)-1)] + [hfile.ref])
        hfile.create_dataset(
            name='dsetobjref',
            shape=shape,
            chunks=(10, 2, 4, 13),
            data=data,
            dtype=h5py.ref_dtype,
            maxshape=(72, 28, 16, 68),
            track_times=None,
            track_order=None,
            fillvalue=None,
            )

        return hfile
