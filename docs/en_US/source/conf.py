# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'HyperGBM'
copyright = '2020, Zetyun.com'
author = 'Zetyun.com'

# The full version, including alpha/beta/rc tags
release = '0.2.2'
extensions = ['recommonmark',
              'sphinxcontrib.mermaid'
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode'
              # 'sphinx.ext.autodoc',
              # 'sphinx.ext.mathjax',
              # 'sphinx.ext.ifconfig',
              # 'sphinx.ext.viewcode',
              # 'sphinx.ext.githubpages',
              ]
exclude_patterns = []
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
html_static_path = ['_static']

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'hypergbm', 'HyperGBM Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'HyperGBM', 'HyperGBM Documentation',
     author, 'HyperGBM', 'One line description of project.',
     'Miscellaneous'),
]
