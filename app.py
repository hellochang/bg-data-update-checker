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
import matplotlib.pyplot as plt
import datetime as dt

from datetime import datetime, timedelta

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
# =============================================================================
# Functions - Analyse Data
# =============================================================================
query_bmc_monthly = """SELECT * FROM development.dbo.bmc_monthly WHERE bm_id IN ('sp500', 'sptsx')"""
query_univsnapshot = """SELECT *
FROM fstest.dbo.univsnapshot
WHERE univ_id IN ('CANADA', 'US')"""
query_portholding = """SELECT *
FROM development.dbo.portholding
WHERE company NOT IN ('Placeholder', 'Cash-INVEST-USD', 'Cash-INVEST-CAD')"""
query_bg_div_ltm = """SELECT *
FROM fstest.dbo.bg_div_ltm"""
query_bg_div = """SELECT fsym_id, exdate
FROM fstest.dbo.bg_div"""
query_portreturn = """SELECT *
FROM development.dbo.PortReturn"""
query_bmprices = """SELECT *
FROM development.dbo.BMPrice"""


## change date format
### allow_output_mutation=True)?
# Perform query and fetch data from SQL server.
@st.cache(allow_output_mutation=True)
def load_data(query):
    data = DataImporter(verbose=False)
    return data.load_data(query)

@st.cache
def find_null(df, col):
    return df[df[col].isnull()]
# Return rows where at least 1 cell in the dataframe is not equal
@st.cache
def df_not_equal(df1, df2):
    df2_sub = df2.iloc[:, 0:2]
    return df2[df1.ne(df2_sub).any(axis=1)]

# Differences in rdate between consec rows
@st.cache
def find_daily(df, group):
    df.loc[:, 'weekday'] =  pd.to_datetime(df['rdate']).dt.weekday
    df.loc[:, 'diff_days'] = df.groupby(group)['rdate'].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
    df.loc[:, 'is_not_daily'] = df.apply(is_not_daily, axis=1)
    return df[df['is_not_daily'] == True]

def is_not_daily(row):
    diff_days = row.diff_days
    # st.write(type(row.rdate))
    # st.write(row.rdate)

    if (row.weekday == 0) and (diff_days == 3):
        return False
    elif diff_days == 1:
        return False
    else:
        return True
    
@st.cache
def find_univsnapshot(df):
    tol = 300  
    df.loc[:, 'monthly_company_count'] = df['univ_id'].groupby(df['rdate'].dt.month).transform('count')
    df.loc[:, 'diff_monthly'] = df['monthly_company_count'].diff()

    return df[df['diff_monthly'] > tol]
@st.cache
def not_in_adjpricest(df):
    adjpricet_fsym_id = adjpricet['fsym_id'].unique()  
    return df[~df['fsym_id'].isin(adjpricet_fsym_id)]

def show_res(res_df):
    if res_df.empty:
        st.success(success_msg)
    else:
        st.error(error_msg)
        st.write(res_df)   
        
# =============================================================================
# Functions - Printing Calendar
# =============================================================================
def show_cal():
    # dates, data = generate_data()
    data = is_in_res_date
    # st.write(dates)
    # st.write(data)
    # st.write(type(dates))
    # st.write(type(data))

    fig, ax = plt.subplots(figsize=(6, 10))
    calendar_heatmap(ax, dates, data)
    fig.tight_layout()

    st.pyplot(fig)

def generate_data():
    num = 100
    data = np.random.randint(0, 20, num)
    start = dt.datetime(2015, 3, 13)
    dates = [start + dt.timedelta(days=i) for i in range(num)]
    return dates, data

def calendar_array(dates, data):
    i, j = zip(*[d.isocalendar()[1:] for d in dates])
    i = np.array(i) - min(i)
    j = np.array(j) - 1
    ni = max(i) + 1

    calendar = np.nan * np.zeros((ni, 7))
    calendar[i, j] = data
    return i, j, calendar


def calendar_heatmap(ax, dates, data):
    i, j, calendar = calendar_array(dates, data)
    im = ax.imshow(calendar, interpolation='none', cmap='summer')
    label_days(ax, dates, i, j, calendar)
    label_months(ax, dates, i, j, calendar)
    # ax.figure.colorbar(im)

def label_days(ax, dates, i, j, calendar):
    ni, nj = calendar.shape
    day_of_month = np.nan * np.zeros((ni, 7))
    day_of_month[i, j] = [d.day for d in dates]

    for (i, j), day in np.ndenumerate(day_of_month):
        if np.isfinite(day):
            ax.text(j, i, int(day), ha='center', va='center')
            
    ax.set(xticks=np.arange(7), 
           xticklabels=['M', 'T', 'W', 'R', 'F', 'S', 'S'])
    ax.xaxis.tick_top()

def label_months(ax, dates, i, j, calendar):
    month_labels = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                             'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    months = np.array([d.month for d in dates])
    uniq_months = sorted(set(months))
    yticks = [i[months == m].mean() for m in uniq_months]
    labels = [month_labels[m - 1] for m in uniq_months]
    ax.set(yticks=yticks)
    ax.set_yticklabels(labels, rotation=90)

        
success_msg = 'No error found.'
error_msg = 'Error rows'

portholding = load_data(query_portholding)
bmprices = load_data(query_bmprices)
portreturn = load_data(query_portreturn)
bmc_monthly = load_data(query_bmc_monthly)
div_ltm = load_data(query_bg_div_ltm)
div = load_data(query_bg_div)
univsnapshot = load_data(query_univsnapshot)

# =============================================================================
# Code
# =============================================================================

# Set page layout
# st.set_page_config(layout="wide")
st.title('Data Quality Checker')
st.text('Select a table to view details.')
in_bmprices = bmprices
in_portreturn = portreturn
in_univsnapshot = univsnapshot
# df.loc[:, 'num_stock'] = df['rdate'].groupby(df['rdate']).transform('count')

query_adjpricet = """SELECT [fsym_id]
  FROM [FSTest].[dbo].[AdjustedPriceTickers]"""
adjpricet = load_data(query_adjpricet)
# Differences in rdate between consec rows


option_sum = 'Summary'
option_date = 'View by Date'
option = st.selectbox('Choose a view', (option_sum, option_date))

res_bmc_monthly = find_null(bmc_monthly, 'fsym_id')
res_portholding = find_null(portholding, 'secid')
res_bmprices = find_daily(in_bmprices, 'bm_id')
res_portreturn = find_daily(in_portreturn, 'pid')
### Univ snapshot can't see the reason for which is which?
res_univsnapshot = find_univsnapshot(in_univsnapshot)
res_univ_notin_id = not_in_adjpricest(in_univsnapshot)
res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
res_div_ltm = df_not_equal(div, div_ltm)
 


   
# def merge_df(df1, df2, df3, df4, df5, df6):
#     df1.set_index('rdate',inplace=True)
#     df2.set_index('rdate',inplace=True)
#     df3.set_index('rdate',inplace=True)
#     df4.set_index('rdate',inplace=True)
#     df5.set_index('date',inplace=True)
#     df6.set_index('rdate',inplace=True)

#     df = pd.concat([df1,df2,df3, df4, df5, df6],axis=1,sort=False).reset_index()
#     df.rename(columns = {'index':'Col1'})
#     return df

# merge_df(res_portholding, res_bmprices, res_portreturn, res_bmc_monthly, res_div_ltm, res_univsnapshot)


## monthly
### univsnap, bmc monthly, div_ltm
if option == 'Summary':
    res_date_lst = [res_portholding['rdate'], res_bmprices['rdate'], res_portreturn['rdate'], 
                    res_bmc_monthly['rdate'], res_div_ltm['date'], res_univsnapshot['rdate']]
    res_date_lst = pd.concat(res_date_lst)
    res_date_lst = res_date_lst.unique()
    # st.write(type(res_date_lst))
    # st.write(res_date_lst)
    
    sdate = st.date_input("Choose start date", 
                          value = datetime(2021, 10, 27))
    edate = st.date_input("Choose end date", 
                          value = datetime(2021, 10, 29))
    
    # input_start_year = st.number_input(
    # 'Select a start year', min_value = 1990, max_value=2021,
    # value=datetime.now().year, format='%d')
    # sdate = datetime(input_start_year, 1, 1)
    # edate = datetime(input_start_year, 12, 31)
    
    dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
    dates = dates.tolist()
    # st.write(dates)
    # st.write(type(dates))
    
    # set(array1) & set(array2)
    is_in_res_date = []
    for elem in dates:
        if elem in res_date_lst:
            is_in_res_date.append(1)
        else:
            is_in_res_date.append(0)
    is_in_res_date = np.asarray(is_in_res_date, dtype=np.int64)

# st.write(is_in_res_date)
# st.write(type(is_in_res_date))  
    show_cal()
    
    st.header('Choose a table to view full result')
    if st.button('BMC Monthly'):
        show_res(res_bmc_monthly)
    if st.button('Portfolio Holding'):
        show_res(res_portholding)
    if st.button('Portfolio Return'):
        show_res(res_portreturn)
    if st.button('BM Price'):
        show_res(res_bmprices)
    if st.button('Universe Snapshot'):
        show_res(res_univsnapshot)
    if st.button('Div LTM'):
        show_res(res_div_ltm)

elif option == 'View by Date':
    input_date = st.date_input("Choose a date")
    res_portholding = portholding[portholding['rdate'] == input_date]
    res_bmprices = bmprices[bmprices['rdate'] == input_date]
    res_portreturn = portreturn[portreturn['rdate'] == input_date]

    # input_date = datetime.combine(input_date, datetime.min.time())
    # st.write(type(input_date))
    # st.write(input_date)
    # st.write(pd.Period(input_date,freq='M').end_time.date())
    # st.write(lday)
    # st.write(type(lday))
    # st.write(type(input_date))
    # st.write(input_date)

    input_date_m = pd.Period(input_date,freq='M').end_time.date()

    res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'] == input_date_m]
    res_div_ltm = div_ltm[div_ltm['date'] == input_date_m]
    res_univsnapshot = univsnapshot[univsnapshot['rdate'] == input_date_m]
    
    

    if st.button('BMC Monthly'):
        show_res(res_bmc_monthly)
    if st.button('Portfolio Holding'):
        show_res(res_portholding)
    if st.button('Portfolio Return'):
        show_res(res_portreturn)
    if st.button('BM Price'):
        show_res(res_bmprices)
    if st.button('Universe Snapshot'):
        show_res(res_univsnapshot)
    if st.button('Div LTM'):
        show_res(res_div_ltm)

