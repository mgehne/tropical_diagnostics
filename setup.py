#!/usr/bin/env python

from setuptools import setup

setup(name='tropical-diagnostics',
      version='0.1',
      # list folders, not files
      packages=['diagnostics',
                'utils'],
      package_data={'diagnostics': ['../data']}, install_requires=['netCDF4', 'numpy', 'xarray', 'scipy',
                                                                   'pandas', 'plotly']
      )
