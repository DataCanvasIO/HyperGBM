# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import tempfile
import os
import time

test_output_dir = tempfile.mkdtemp(prefix=time.strftime("hygbm_test_%m%d%H%M_"))
os.environ['DEEPTABLES_HOME'] = test_output_dir
os.environ['HYPERNETS_HOME'] = test_output_dir
