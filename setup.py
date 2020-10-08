#!/usr/bin/env python

from setuptools import setup

setup(name='tropical_diagnostics',
      version='0.1',
      # list folders, not files
      packages=['tropical_diagnostics'],
      package_data={'tropical_diagnostics': ['../data']}, install_requires=['netCDF4', 'numpy', 'xarray', 'scipy',
                                                                   'pandas', 'plotly', 'kaleido', 'matplotlib']
      )
