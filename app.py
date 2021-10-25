# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:13:47 2021

@author: Chang.Liu
"""
# =============================================================================
# Import packages
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
# =============================================================================
# Functions - Analyse Data
# =============================================================================
query_bmc_monthly = """SELECT * FROM development.dbo.bmc_monthly AND bm_id IN ('sp500', 'sptsx')"""
query_univsnapshot = """SELECT *
FROM fstest.dbo.univsnapshot"""
query_portholding = """SELECT *
FROM development.dbo.portholding
WHERE cusip NOT LIKE 'cash'"""
query_bg_div_ltm = """ """
query_portreturn = """SELECT *
FROM development.dbo.PortReturn"""

query_bmprices = """SELECT *
FROM development.dbo.BMPrice"""


## change date format

# Perform query and fetch data from SQL server.
@st.cache
def load_data(query):
    data = DataImporter(verbose=False)
    return data.load_data(query)

def find_null(df, col):
    return df[df[col].isnull()]

portholding = load_data(query_portholding)
bmprices = load_data(query_bmprices)
portreturn = load_data(query_portreturn)
find_null()
find_null(portholding, 'secid')

# df.loc[:, 'num_stock'] = df['rdate'].groupby(df['rdate']).transform('count')

def find_daily(df, group):
    df.loc[:, 'diff_days'] = df.groupby(group)['rdate'].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
    return df[df['diff_days'] != 1]

find_daily(bmprices, 'bm_id')
find_daily(portreturn, 'pid')
