# -*- coding:utf-8 -*-

from __future__ import absolute_import

import os
from os import path as P

from setuptools import find_packages
from setuptools import setup

my_name = 'hypergbm'
home_url = 'https://github.com/DataCanvasIO/HyperGBM'


def read_requirements(file_path='requirements.txt'):
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


def read_description(file_path='README.md',
                     image_root=f'{home_url}/raw/main',
                     embed_size_limit=1024 * 120):
    import re
    import os

    def _encode_image(m):
        assert len(m.groups()) == 3

        pre, src, post = m.groups()
        src = src.rstrip().lstrip()

        # if embed_size_limit is not None and embed_size_limit > os.path.getsize(src):
        #     import base64
        #     ext = src[src.rfind('.') + 1:].lower()
        #     data = open(src, 'rb').read()
        #     txt = base64.b64encode(data).decode()
        #     return f'{pre}data:image/{ext};base64,{txt}{post}'
        # else:
        #     remote_src = os.path.join(image_root, src)
        #     return f'{pre}{remote_src}{post}'
        remote_src = os.path.join(image_root, os.path.relpath(src))
        return f'{pre}{remote_src}{post}'

    desc = open(file_path, encoding='utf-8').read()

    # substitute html image
    desc = re.sub(r'(<img\s+src\s*=\s*\")(docs/static/images/[^"]+)(\")', _encode_image, desc)

    # substitute markdown image
    desc = re.sub(r'(\!\[.*\]\()(docs/static/images/.+)(\))', _encode_image, desc)

    return desc


try:
    execfile
except NameError:
    def execfile(fname, globs, locs=None):
        locs = locs or globs
        exec(compile(open(fname).read(), fname, "exec"), globs, locs)

HERE = P.dirname((P.abspath(__file__)))

version_ns = {}
execfile(P.join(HERE, 'hypergbm', '_version.py'), version_ns)
version = version_ns['__version__']

print("__version__=" + version)

MIN_PYTHON_VERSION = '>=3.6'

# long_description = open('README.md', encoding='utf-8').read()
long_description = read_description('README.md')

requires = read_requirements()
extras_require = read_extra_requirements()

setup(
    name=my_name,
    version=version,
    description='A full pipeline AutoML tool integrated various GBM models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=home_url,
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests')),
    package_data={
        'hypergbm': ['examples/*', 'examples/**/*', 'examples/**/**/*'],
    },
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hypergbm = hypergbm.utils.tool:main',
        ],
    },
    include_package_data=True,
)
