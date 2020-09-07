from setuptools import setup, find_packages

install_requires = ['numpy',
                    'cython',
                    'pkgconfig',
                    'zarr>=2.4.0',
                    'numcodecs',
                    'h5py @ git+https://github.com/h5py/h5py.git',
                    'fsspec',
                    's3fs',
                    'hdmf',
                    'nwbwidgets']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="HDF5Zarr",
    version=0.1,
    url="https://github.com/catalystneuro/HDF5Zarr",
    description="Reading HDF5 files with Zarr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Sotoude, Ben Dichter",
    author_email="dsot@protonmail.com, ben.dichter@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=['Operating System :: OS Independent',
                 'Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: BSD License',
                 'Programming Language :: Python :: 3.8',
                 ],
)
