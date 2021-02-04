# -*- coding:utf-8 -*-

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup


def read_requirements(file_path='requirements.txt'):
    import os

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r')as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

    return lines


def read_extra_requirements():
    import glob
    import re

    extra = {}

    for file_name in glob.glob('requirements-*.txt'):
        key = re.search('requirements-(.+).txt', file_name).group(1)
        req = read_requirements(file_name)
        if req:
            extra[key] = req

    if extra and 'all' not in extra.keys():
        extra['all'] = sorted({v for req in extra.values() for v in req})

    return extra


import hypergbm

version = hypergbm.__version__

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

requires = read_requirements()
extras_require = read_extra_requirements()

setup(
    name='hypergbm',
    version=version,
    description='A full pipeline AutoML tool integrated various GBM models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='DataCanvas Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requires,
    python_requires=MIN_PYTHON_VERSION,
    extras_require=extras_require,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests')),
    package_data={
    },
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hypergbm = hypergbm.utils.cli:main',
        ]
    },
    include_package_data=True,
)
