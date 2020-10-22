# -*- coding:utf-8 -*-
import pandas as pd
import os

basedir = os.path.dirname(__file__)


def load_bank():
    data = pd.read_csv(f'{basedir}/bank-uci.csv')
    return data
