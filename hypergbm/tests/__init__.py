# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import os
import tempfile
import time

from hypernets.tabular.cfg import TabularCfg

TabularCfg.tfidf_primitive_output_feature_count = 5

test_output_dir = tempfile.mkdtemp(prefix=time.strftime("hygbm_test_%m%d%H%M_"))

os.environ['DEEPTABLES_HOME'] = test_output_dir
os.environ['HYPERNETS_HOME'] = test_output_dir
