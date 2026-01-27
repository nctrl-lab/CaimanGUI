#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for CaImAnGUI
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='caimangui',
    version='1.1.0',
    description='A graphical user interface for visualizing CaImAn processed imaging data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hung-Ling',
    url='https://github.com/nctrl-lab/CaImAnGUI',
    license='GPL-3.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pyqtgraph',
        'scipy',
        'matplotlib',
        'PyQt5',
        'python-dateutil',
        'pynwb',
        'caiman',
        'natsort',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'caiman=gui.caiman_gui:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    include_package_data=True,
    package_data={
        'gui': ['config_nwb.json'],
    },
)
