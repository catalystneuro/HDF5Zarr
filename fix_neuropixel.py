import os

from h5py import File
from uuid import uuid4


def fix_pynwb_read(fpath):

    with File(fpath, 'r+') as f:
        # remove second row of timestamps
        dataset_paths = ['/processing/running/running_speed/timestamps',
                         '/processing/running/running_wheel_rotation/timestamps']
        for dataset_path in dataset_paths:

            # read dataset
            attrs = dict(f[dataset_path].attrs)
            dset_location, dset_name = os.path.split(dataset_path)
            try:
                # there's a weird error where this throws an exception even though it works as expected.
                # I'm using a try/except to get around it
                data = f[dataset_path][0, :]
            except TypeError:
                data = f[dataset_path][:]

            # delete
            del f[dataset_path]

            # write new dataset
            f[dset_location].create_dataset(dset_name, data=data)
            for key, val in attrs.items():
                f[dataset_path].attrs[key] = val

        # rename the "name" column to "type"
        group_path = '/processing/optotagging/optogenetic_stimuluation'

        # change value in colnames attribute
        colnames = f[group_path].attrs['colnames']
        colnames[colnames == b'name'] = b'type'
        f[group_path].attrs['colnames'] = colnames

        # read and write dataset
        name_dset_path = os.path.join(group_path, 'name')
        label_dset_path = os.path.join(group_path, 'type')

        f[label_dset_path] = f[name_dset_path]

        # delete old dataset
        del f[name_dset_path]


def write_subject(fpath):
    with File(fpath, 'r+') as f:

        metadata = dict(f['/general/metadata'].attrs)
        dsets = dict(
            sex=metadata['sex'],
            age='{} days'.format(metadata['age_in_days']),
            species='Mus musculus',
            genotype=metadata['full_genotype'],
            subject_id=metadata['specimen_name']
        )

        attributes = dict(
            namespace='core',
            neurodata_type='Subject',
            object_id=str(uuid4()),
            description='strain: {}'.format(metadata['strain'])
        )

        f['/general'].create_group('subject')
        for key, val in attributes.items():
            f['general/subject'].attrs[key] = val

        for key, val in dsets.items():
            f['general/subject'].create_dataset(key, data=val)


fpath = '/Volumes/easystore5T/data/Allen/neuropixel/fixed/ecephys_session_756029989.nwb'

fix_pynwb_read(fpath)
write_subject(fpath)

# test read
from pynwb import NWBHDF5IO
import allensdk.brain_observatory.ecephys.nwb

with NWBHDF5IO(fpath, 'r') as f:
    nwb = f.read()

print('done!')
