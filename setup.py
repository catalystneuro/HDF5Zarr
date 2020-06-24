from setuptools import setup, find_packages

with open('requirements.txt', "r") as f:
    install_requires = f.read().split()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="allen-institute-neuropixel-utils",
    version=0.1,
    url="https://github.com/catalystneuro/allen-institute-neuropixel-utils",
    description="allen-institute-neuropixel-utils",
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
