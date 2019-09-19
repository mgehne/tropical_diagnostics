#!/usr/bin/env python

from setuptools import setup

setup(name='tropical_diagnostics',
      version='1.0',
      # list folders, not files
      packages=['diagnostics',
                'utils'],
      scripts=['examples/*.py'],
      package_data={'diagnostics': ['data/*.nc']},
      )
