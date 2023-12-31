#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup
from hermesv3_bu import __version__


# Get the version number from the relevant file
version = __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='hermesv3_bu',
    # license='',
    # platforms=['GNU/Linux Debian'],
    version=version,
    description='HERMESv3 Bottom-Up',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Carles Tena Medina',
    author_email='carles.tena@bsc.es',
    url='https://earth.bsc.es/gitlab/es/hermesv3_bu',

    keywords=['emissions', 'cmaq', 'monarch', 'wrf-chem', 'atmospheric composition', 'air quality', 'earth science'],
    install_requires=[
        'numpy',
        'netCDF4>=1.3.1',
        'cdo>=1.3.3',
        'pandas',
        'fiona',
        'Rtree',
        'geopandas',
        'pyproj',
        'configargparse',
        'cf_units>=1.1.3',
        'pytz',
        'timezonefinder',
        'mpi4py',
        'shapely',
        'rasterio',
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ],
    package_data={'': [
        'README.md',
        'CHANGELOG',
        'LICENSE',
    ]
    },

    entry_points={
        'console_scripts': [
            'hermesv3_bu = hermesv3_bu.hermes:run',
            'hermesv3_bu_download_benchmark = hermesv3_bu.tools.download_benchmark:download_benchmark',
        ],
    },
)