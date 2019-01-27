# -*- coding: utf-8 -*-
# Copyright (c) 2018 Luca Pinello
# Made available under the MIT license.


#This is based on: https://github.com/Roastero/freshroastsr700

import os
from setuptools import setup
from setuptools import find_packages


here = os.path.abspath(os.path.dirname(__file__))


setup(
    name='SR700_Artisan_Profile_Builder',
    version=0.2,
    description='Create artisan profiles using the alarms and the modified version of the freshroastsr700 library for phidget',
    url='https://github.com/lucapinello/create_artisan_alarms_phidget',
    author='Luca Pinello',
    license='GPLv3',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'PySide2',
    ],
    entry_points = {'console_scripts': ['SR700_Artisan_Profile_Builder=SR700_Artisan_Profile_Builder:main']}
    )
