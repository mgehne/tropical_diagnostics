#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.1.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open('README.rst',encoding='utf-8') as readme_file:
    readme = readme_file.read()

# Get the history from the HISTORY file
with open('HISTORY.rst',encoding='utf-8') as history_file:
    history = history_file.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    version=__version__,
    name='wk_spectra',
    description='A Python package for the construction of the Wheeler-Kiladis Space-Time Spectra.',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author='Alejandro Jaramillo',
    author_email='ajaramillomoreno@gmail.com',
    url='https://github.com/ajaramillomoreno/wk_spectra',
    download_url='https://github.com/ajaramillomoreno/wk_spectra/tarball/' + __version__,
    license="BSD license",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    keywords= ['wk_spectra','CCEWs'],
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
    zip_safe=False
)
