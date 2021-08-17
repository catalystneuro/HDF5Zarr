import pytest
import os
from shutil import copy
from test_hdf5zarr import HDF5ZarrBase


def pytest_addoption(parser):
    parser.addoption(
        "--hdf5files",
        action="append",
        type=str,
        default=[],
        help="list of hdf5 files to test",
    )
    parser.addoption(
        "--numfiles",
        action="store",
        type=int,
        default=3,
        help="number of temporary test hdf5 files to create, ignored if --hdf5files is passed",
    )
    parser.addoption(
        "--disablemaxchunk",
        action="store_true",
        help="flag to disable testing max_chunksize argument",
    )
    parser.addoption(
        "--numsubgroup",
        action="store",
        type=int,
        default=4,
        help="number of runs testing hdf5group/hdf5obj argument",
    )
    parser.addoption(
        "--numdataset",
        action="store",
        type=int,
        default=1,
        help="number of runs testing h5py.Dataset as filename argument",
    )
    parser.addoption(
        "--fkeep",
        action="store_true",
        help="flag to indicate collecting failed objects",
    )
    parser.addoption(
        "--objnames",
        action="append",
        type=str,
        default=[],
        help="name of objects in file to test",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    # accessed in fixtures
    setattr(item, "rep_" + rep.when, rep)


def pytest_generate_tests(metafunc):
    hdf5files = metafunc.config.getoption('hdf5files')
    disable_max_chunksize = metafunc.config.getoption('disablemaxchunk')
    numsubgroup = metafunc.config.getoption('numsubgroup')
    numdset = metafunc.config.getoption('numdataset')
    objnames = metafunc.config.getoption('objnames')
    cls = metafunc.cls

    if len(hdf5files) != 0:
        cls.hdf5files_option = True
    else:
        cls.hdf5files_option = False
        numfiles = metafunc.config.getoption('numfiles')
        # test file names for _create_file
        hdf5files = [f"file{i}" for i in range(numfiles)]

    cls.hdf5file_names = hdf5files
    cls.disable_max_chunksize = disable_max_chunksize
    cls.numsubgroup = numsubgroup
    cls.numdset = numdset
    cls.objnames = [n.encode() for n in objnames]
    cls._testfilename = "_testfile"  # test file name for _testfile, only used if hdf5files_option is False

    cls.ids_testfilename = [cls._testfilename]*int(not cls.hdf5files_option)
    cls.ids_hdf5files = hdf5files
    ids = cls.ids_testfilename + cls.ids_hdf5files
    cls.ids_subgroup = [i+'-subgroup' for i in ids]*numsubgroup
    cls.ids_dset = [i+'-dataset' for i in ids]*numdset
    ids += cls.ids_subgroup + cls.ids_dset
    cls.num_maxchunksize = 2
    cls.ids_maxchunksize = [i+'-maxchunksize' for i in ids]*int(not disable_max_chunksize)*cls.num_maxchunksize
    ids += cls.ids_maxchunksize
    metafunc.module.confargs = (cls.hdf5files_option, cls.hdf5file_names, cls.ids_subgroup,
                                cls.ids_dset, cls.ids_maxchunksize, cls._testfilename)
    if metafunc.definition.nodeid.rfind('TestHDF5Zarr') > 0:
        metafunc.fixturenames.append('fnum')
        metafunc.parametrize(argnames='fnum', argvalues=range(len(ids)), ids=ids, indirect=True)


def pytest_runtest_teardown(item, nextitem):
    if item.rep_setup.passed and item.rep_call.failed:
        fnum = item.callspec.params['fnum']//item.instance.num_files
        if not item.instance.fnum_keep[fnum]:
            copy(item.instance.hfile.filename, os.getcwd())
            item.instance.fnum_keep[fnum] = True

    item.session._setupstate.teardown_exact(item, nextitem)


@pytest.fixture(scope="session")
def filesbase(request):
    args = request.session.items[0].module.confargs
    if all([item.name.rfind('open_as_zarr_dset') > 0 for item in request.session.items]):
        # for open_as_zarr_dset disable subgroups, subdataset and max_chunksize options
        hdf5files_option, hdf5file_names, _, _, _, _testfilename = *args,
        base = HDF5ZarrBase(hdf5files_option, hdf5file_names, [], [], [], _testfilename)
    else:
        base = HDF5ZarrBase(*args)
    return base
