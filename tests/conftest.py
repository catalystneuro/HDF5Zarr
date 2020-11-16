import pytest
import os
from shutil import copy


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
    objnames = metafunc.config.getoption('objnames')

    metafunc.fixturenames.append('fnum')
    if len(hdf5files) != 0:
        metafunc.cls.hdf5files_option = True
    else:
        metafunc.cls.hdf5files_option = False
        numfiles = metafunc.config.getoption('numfiles')
        # test file names for _create_file
        hdf5files = [f"file{i}" for i in range(numfiles)]

    metafunc.cls.hdf5file_names = hdf5files
    metafunc.cls.disable_max_chunksize = disable_max_chunksize
    metafunc.cls.objnames = [n.encode() for n in objnames]
    metafunc.cls._testfilename = "_testfile"  # test file name for _testfile, only used if hdf5files_option is False

    metafunc.parametrize(argnames='fnum', argvalues=range(len(hdf5files)+int(not metafunc.cls.hdf5files_option)),
                         ids=[metafunc.cls._testfilename]*int(not metafunc.cls.hdf5files_option)+hdf5files, indirect=True)


def pytest_runtest_teardown(item, nextitem):
    if item.rep_setup.passed and item.rep_call.failed:
        fnum = item.callspec.params['fnum']
        if not item.instance.fnum_keep[fnum]:
            copy(item.instance.hfile.filename, os.getcwd())
            item.instance.fnum_keep[fnum] = True

    item.session._setupstate.teardown_exact(item, nextitem)
