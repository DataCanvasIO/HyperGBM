# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import tempfile
import os

test_output_dir = tempfile.mkdtemp()
os.environ['DEEPTABLES_HOME'] = test_output_dir
os.environ['HYPERNETS_HOME'] = test_output_dir
