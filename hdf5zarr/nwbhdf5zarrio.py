import zarr
from collections.abc import MutableMapping
from collections import deque
from functools import partial
from hdmf.utils import call_docval_func, docval, popargs, getargs
from hdmf.backends.hdf5 import HDF5IO as _HDF5IO
from hdmf.build import BuildManager, TypeMap, GroupBuilder, LinkBuilder, DatasetBuilder, ReferenceBuilder
from hdmf.query import HDMFDataset
from hdmf.query import Array as HDMFArray
from hdmf.backends.hdf5.h5_utils import BuilderH5ReferenceDataset, BuilderH5TableDataset, H5SpecReader
from hdmf.backends.warnings import BrokenLinkWarning
from hdmf.backends.io import UnsupportedOperation
from hdmf.spec import NamespaceCatalog
from pynwb import get_manager, get_type_map
import os
from warnings import warn
import logging

HDMFDataset.register(zarr.Array)

from .hdf5zarr import SYMLINK
ROOT_NAME = 'root'
SPEC_LOC_ATTR = '.specloc'


class ZarrSpecReader(H5SpecReader):

    @docval({'name': 'group', 'type': zarr.Group, 'doc': 'the zarr Group to read specs from'})
    def __init__(self, **kwargs):
        group = getargs('group', kwargs)
        self._H5SpecReader__group = group
        super_kwargs = {'source': "%s:%s" % (os.path.abspath(group.file.name), group.name)}
        call_docval_func(super(H5SpecReader, self).__init__, super_kwargs)
        self._H5SpecReader__cache = None


class NWBZARRHDF5IO(_HDF5IO):

    @docval({'name': 'path', 'type': str, 'doc': 'the path to the HDF5 file', 'default': None},
            {'name': 'mode', 'type': str,
             'doc': 'the mode to open the HDF5 file with, one of ("w", "r", "r+", "a", "w-", "x")'},
            {'name': 'load_namespaces', 'type': bool,
             'doc': 'whether or not to load cached namespaces from given path - not applicable in write mode',
             'default': False},
            {'name': 'manager', 'type': BuildManager, 'doc': 'the BuildManager to use for I/O', 'default': None},
            {'name': 'extensions', 'type': (str, TypeMap, list),
             'doc': 'a path to a namespace, a TypeMap, or a list consisting paths to namespaces and TypeMaps',
             'default': None},
            {'name': 'file', 'type': MutableMapping, 'doc': 'a pre-existing MutableMapping object', 'default': None},
            {'name': 'comm', 'type': "Intracomm", 'doc': 'the MPI communicator to use for parallel I/O',
             'default': None},
            enforce_type=False, allow_extra=True)
    def __init__(self, **kwargs):
        path, mode, manager, extensions, load_namespaces, file_obj, comm =\
            popargs('path', 'mode', 'manager', 'extensions', 'load_namespaces', 'file', 'comm', kwargs)

        # root group
        self.__rgroup = file_obj
        chunk_store = getattr(file_obj, 'chunk_store', None)
        if chunk_store is not None:
            try:
                filename = getattr(chunk_store.source, 'path', None)
                if filename is None:
                    filename = chunk_store.source.name
            except:
                filename = None
        if filename is None:
            filename = f'{type(file_obj.store).__name__}'
        self.__rgroup.filename = filename

        file_obj = self.__set_rgroup(file_obj)

        self.__built = dict()       # keep track of each builder for each dataset/group/link for each file
        self.__read = dict()        # keep track of which files have been read. Key is the filename value is the builder
        self.__file = file_obj

        if load_namespaces:
            if manager is not None:
                warn("loading namespaces from file - ignoring 'manager'")
            if extensions is not None:
                warn("loading namespaces from file - ignoring 'extensions' argument")
            # namespaces are not loaded when creating an NWBZARRHDF5IO object in write mode
            if 'w' in mode or mode == 'x':
                raise ValueError("cannot load namespaces from file when writing to it")

            tm = get_type_map()
            self.load_namespaces(tm, path, file=file_obj)
            manager = BuildManager(tm)

            # XXX: Leaving this here in case we want to revert to this strategy for
            #      loading cached namespaces
            # ns_catalog = NamespaceCatalog(NWBGroupSpec, NWBDatasetSpec, NWBNamespace)
            # super(NWBZARRHDF5IO, self).load_namespaces(ns_catalog, path)
            # tm = TypeMap(ns_catalog)
            # tm.copy_mappers(get_type_map())
        else:
            if manager is not None and extensions is not None:
                raise ValueError("'manager' and 'extensions' cannot be specified together")
            elif extensions is not None:
                manager = get_manager(extensions=extensions)
            elif manager is None:
                manager = get_manager()

        self.logger = logging.getLogger('%s.%s' % (self.__class__.__module__, self.__class__.__qualname__))

        if file_obj is not None:
            if path is None:
                path = file_obj.filename
            elif os.path.abspath(file_obj.filename) != os.path.abspath(path):
                msg = 'You argued %s as this object\'s path, ' % path
                msg += 'but supplied a file with filename: %s' % file_obj.filename
                raise ValueError(msg)
        elif path is None:
            TypeError("Must supply either 'path' or 'file' arg to HDF5IO.")

        if file_obj is None and not os.path.exists(path) and (mode == 'r' or mode == 'r+'):
            msg = "Unable to open file %s in '%s' mode. File does not exist." % (path, mode)
            raise UnsupportedOperation(msg)

        if file_obj is None and os.path.exists(path) and (mode == 'w-' or mode == 'x'):
            msg = "Unable to open file %s in '%s' mode. File already exists." % (path, mode)
            raise UnsupportedOperation(msg)

        if manager is None:
            manager = BuildManager(TypeMap(NamespaceCatalog()))
        elif isinstance(manager, TypeMap):
            manager = BuildManager(manager)

        # TO DO #
        self._HDF5IO__comm = comm
        self._HDF5IO__mode = mode
        self._HDF5IO__path = path
        self._HDF5IO__file = file_obj
        super(_HDF5IO, self).__init__(manager, source=path)
        self._HDF5IO__ref_queue = deque()  # a queue of the references that need to be added
        self._HDF5IO__dci_queue = deque()  # a queue of DataChunkIterators that need to be exhausted

    @docval(returns='a GroupBuilder representing the data object', rtype='GroupBuilder')
    def read_builder(self):
        f_builder = self.__read.get(self.__file.name)
        # ignore cached specs when reading builder
        ignore = set()
        specloc = self.__file.attrs.get(SPEC_LOC_ATTR)
        if specloc is not None:
            ignore.add(self.__file[specloc].name)
        if f_builder is None:
            f_builder = self.__read_group(self.__file, ROOT_NAME, ignore=ignore)
            self.__read[self.__file.name] = f_builder
        return f_builder

    @docval({'name': 'namespace_catalog', 'type': (NamespaceCatalog, TypeMap),
             'doc': 'the NamespaceCatalog or TypeMap to load namespaces into'},
            {'name': 'path', 'type': str, 'doc': 'the path to the zarr Group', 'default': None},
            {'name': 'namespaces', 'type': list, 'doc': 'the namespaces to load', 'default': None},
            {'name': 'file', 'type': zarr.Group, 'doc': 'a pre-existing zarr.Group object', 'default': None},
            returns="dict with the loaded namespaces", rtype=dict)
    def load_namespaces(self, **kwargs):
        '''
        Load cached namespaces from a file.
        '''

        namespace_catalog, path, namespaces, file_obj = popargs('namespace_catalog', 'path', 'namespaces', 'file',
                                                                kwargs)

        if file_obj is None:
            raise ValueError("'file' argument must be supplied to load_namespaces.")

        if path is not None and file_obj is not None:  # consistency check
            if os.path.abspath(file_obj.filename) != os.path.abspath(path):
                msg = ("You argued '%s' as this object's path, but supplied a file with filename: %s"
                       % (path, file_obj.filename))
                raise ValueError(msg)

        return self.__load_namespaces(namespace_catalog, namespaces, file_obj)

    def __load_namespaces(self, namespace_catalog, namespaces, file_obj):
        d = {}

        if SPEC_LOC_ATTR not in file_obj.attrs:
            msg = "No cached namespaces found in %s" % file_obj.filename
            warn(msg)
            return d

        spec_group = file_obj[file_obj.attrs[SPEC_LOC_ATTR]]

        if namespaces is None:
            namespaces = list(spec_group.keys())

        readers = dict()
        deps = dict()
        for ns in namespaces:
            ns_group = spec_group[ns]
            # NOTE: by default, objects within groups are iterated in alphanumeric order
            version_names = list(ns_group.keys())
            if len(version_names) > 1:
                # prior to HDMF 1.6.1, extensions without a version were written under the group name "unversioned"
                # make sure that if there is another group representing a newer version, that is read instead
                if 'unversioned' in version_names:
                    version_names.remove('unversioned')
            if len(version_names) > 1:
                # as of HDMF 1.6.1, extensions without a version are written under the group name "None"
                # make sure that if there is another group representing a newer version, that is read instead
                if 'None' in version_names:
                    version_names.remove('None')
            latest_version = version_names[-1]
            ns_group = ns_group[latest_version]
            ns_group = self.__set_rgroup(ns_group)
            reader = ZarrSpecReader(ns_group)
            readers[ns] = reader
            for spec_ns in reader.read_namespace('namespace'):
                deps[ns] = list()
                for s in spec_ns['schema']:
                    dep = s.get('namespace')
                    if dep is not None:
                        deps[ns].append(dep)

        order = self._order_deps(deps)
        for ns in order:
            reader = readers[ns]
            d.update(namespace_catalog.load_namespaces('namespace', reader=reader))

        return d

    def __set_rgroup(self, obj):
        obj.file = self.__rgroup
        obj.id = obj.name
        if isinstance(obj, zarr.Array):
            obj.maxshape = obj.shape
        return obj

    def __set_built(self, fpath, id, builder):
        """
        Update self.__built to cache the given builder for the given file and id.

        :param fpath: Path to the HDF5 file containing the object
        :type fpath: str
        :param id: ID of the HDF5 object in the path
        :type id: h5py GroupID object
        :param builder: The builder to be cached
        """
        self.__built.setdefault(fpath, dict()).setdefault(id, builder)

    def __get_built(self, fpath, id):
        """
        Look up a builder for the given file and id in self.__built cache

        :param fpath: Path to the HDF5 file containing the object
        :type fpath: str
        :param id: ID of the HDF5 object in the path
        :type id: h5py GroupID object

        :return: Builder in the self.__built cache or None
        """

        fdict = self.__built.get(fpath)
        if fdict:
            return fdict.get(id)
        else:
            return None

    def __read_group(self, h5obj, name=None, ignore=set()):
        kwargs = {
            "attributes": self.__read_attrs(h5obj),
            "groups": dict(),
            "datasets": dict(),
            "links": dict()
        }

        if name is None:
            name = str(os.path.basename(h5obj.name))
        for k in h5obj:
            sub_h5obj = h5obj.get(k)
            sub_h5obj = self.__set_rgroup(sub_h5obj)

            if not (sub_h5obj is None):
                if sub_h5obj.name in ignore:
                    continue

                # TO DO #
                if isinstance(sub_h5obj, zarr.Group) and len(sub_h5obj) == 1 and SYMLINK in sub_h5obj:
                    # Reading links might be better suited in its own function
                    # get path of link (the key used for tracking what's been built)

                    target_path = sub_h5obj[SYMLINK].attrs[sub_h5obj.name]
                    builder_name = os.path.basename(target_path)
                    parent_loc = os.path.dirname(target_path)
                    # get builder if already read, else build it
                    target_obj = sub_h5obj.file[target_path]
                    target_obj = self.__set_rgroup(target_obj)

                    builder = self.__get_built(sub_h5obj.file.filename, target_obj.id)
                    if builder is None:
                        # NOTE: all links must have absolute paths
                        if isinstance(sub_h5obj, zarr.Array):
                            # TO DO #
                            builder = self.__read_dataset(sub_h5obj, builder_name)
                        else:
                            builder = self.__read_group(sub_h5obj, builder_name, ignore=ignore)
                        self.__set_built(sub_h5obj.file.filename, target_obj.id, builder)
                    builder.location = parent_loc
                    link_builder = LinkBuilder(builder, k, source=h5obj.file.filename)
                    link_builder.written = True
                    kwargs['links'][builder_name] = link_builder
                else:
                    builder = self.__get_built(sub_h5obj.file.filename, sub_h5obj.id)
                    obj_type = None
                    read_method = None
                    if isinstance(sub_h5obj, zarr.Array):
                        read_method = self.__read_dataset
                        obj_type = kwargs['datasets']
                    else:
                        read_method = partial(self.__read_group, ignore=ignore)
                        obj_type = kwargs['groups']
                    if builder is None:
                        builder = read_method(sub_h5obj)
                        self.__set_built(sub_h5obj.file.filename, sub_h5obj.id, builder)
                    obj_type[builder.name] = builder
            else:
                warn(os.path.join(h5obj.name, k), BrokenLinkWarning)
                kwargs['datasets'][k] = None
                continue

        kwargs['source'] = h5obj.file.filename
        ret = GroupBuilder(name, **kwargs)
        ret.written = True
        return ret

    def __read_dataset(self, h5obj, name=None):

        h5obj_maxshape = h5obj.shape
        kwargs = {
            "attributes": self.__read_attrs(h5obj),
            "dtype": h5obj.dtype,
            "maxshape": h5obj_maxshape
        }

        if name is None:
            name = str(os.path.basename(h5obj.name))
        kwargs['source'] = h5obj.file.filename
        ndims = len(h5obj.shape)
        if ndims == 0:                                       # read scalar
            scalar = h5obj[()]
            if isinstance(scalar, bytes):
                scalar = scalar.decode('UTF-8')

            # TO DO Reference #
            deref_obj = None
            if isinstance(scalar, str) and scalar != '':
                try:
                    deref_obj = h5obj.file[scalar]
                except:
                    pass
            if deref_obj is not None:
                # TODO (AJTRITT):  This should call __read_ref to support Group references
                target = deref_obj
                target = self.__set_rgroup(target)
                target_builder = self.__read_dataset(target)
                self.__set_built(target.file.filename, target.id, target_builder)
                # TO DO Region Reference #

                # TO DO #
                kwargs['data'] = ReferenceBuilder(target_builder)
            else:
                kwargs["data"] = scalar
        elif ndims == 1:
            d = None
            if h5obj.dtype.kind == 'O' and len(h5obj) > 0:
                elem1 = h5obj[0]
                if isinstance(elem1, (str, bytes)):
                    d = h5obj
            # TO DO #
            elif h5obj.dtype == 'uint64' and len(h5obj) > 0:
                d = BuilderH5ReferenceDataset(HDMFArray(h5obj), self)  # read list of references
                # TO DO Region Reference #
            elif h5obj.dtype.kind == 'V':    # table
                cpd_dt = h5obj.dtype
                # TO DO check_dtype #
                ref_cols = [cpd_dt[i] == 'uint64' for i in range(len(cpd_dt))]
                d = BuilderH5TableDataset(HDMFArray(h5obj), self, ref_cols)
            else:
                d = h5obj
            kwargs["data"] = d
        else:
            kwargs["data"] = h5obj
        ret = DatasetBuilder(name, **kwargs)
        ret.written = True
        return ret

    def __read_attrs(self, h5obj):
        ret = dict()
        for k, v in h5obj.attrs.items():
            if k == SPEC_LOC_ATTR:     # ignore cached spec
                continue
            if isinstance(v, str) and v.startswith('//'):
                try:
                    deref_obj = h5obj.file[v]
                except:
                    deref_obj = None
                if deref_obj is not None:
                    ret[k] = self.__read_ref(deref_obj)
                else:
                    ret[k] = v
            else:
                ret[k] = v
        return ret

    def __read_ref(self, h5obj):

        h5obj = self.__set_rgroup(h5obj)

        ret = None
        ret = self.__get_built(h5obj.file.filename, h5obj.id)
        if ret is None:
            if isinstance(h5obj, zarr.Array):
                ret = self.__read_dataset(h5obj)
            elif isinstance(h5obj, zarr.Group):
                ret = self.__read_group(h5obj)
            else:
                raise ValueError("h5obj must be a Array or a Group - got %s" % str(h5obj))
            self.__set_built(h5obj.file.filename, h5obj.id, ret)
        return ret

    def close(self):
        pass
