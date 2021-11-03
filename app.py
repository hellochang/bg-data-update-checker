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
query_bmc_monthly = """SELECT * FROM development.dbo.bmc_monthly WHERE bm_id IN ('sp500', 'sptsx')"""
query_univsnapshot = """SELECT *
FROM fstest.dbo.univsnapshot
WHERE univ_id IN ('CANADA', 'US')"""
query_portholding = """SELECT *
FROM development.dbo.portholding
WHERE company NOT IN ('Placeholder', 'Cash-INVEST-USD', 'Cash-INVEST-CAD')"""
query_div_ltm = """SELECT div.fsym_id, exdate, date, div_type, div_freq
FROM fstest.dbo.bg_div AS div
LEFT JOIN fstest.dbo.bg_div_ltm AS ltm
ON div.exdate = ltm.date
AND div.fsym_id = ltm.fsym_id
WHERE  div_type ='regular'
AND dummy_payment = 0"""
query_portreturn = """SELECT *
FROM development.dbo.PortReturn"""
query_bmprices = """SELECT *
FROM development.dbo.BMPrice"""

query_holiday = """SELECT [fref_exchange_code]
      ,[holiday_date]
      ,[holiday_name]
  FROM [FSTest].[ref_v2].[ref_calendar_holidays]
WHERE fref_exchange_code IN ('NYS', 'TSE')"""

## change date format
### allow_output_mutation=True)?
# Perform query and fetch data from SQL server.
@st.cache(allow_output_mutation=True)
def load_data(query):
    data = DataImporter(verbose=False)
    return data.load_data(query)



# =============================================================================
# Functions - Check Data Quality
# =============================================================================

@st.cache
def find_null(df, col):
    return df[df[col].isnull()]

# Return True if the dates differ more or less than 1. Helper for find_daily(df, group)
def is_not_daily(row):
    diff_days = row.diff_days
    
    # Check for weekends and remove Mondays
    if (row.weekday == 0) and (diff_days == 3):
        return False
    # For weekdays
    elif diff_days == 1:
        return False
    else:
        return True

# Differences in rdate between consec rows
@st.cache
def find_daily(df, group):
    # df = df.copy()
    df.loc[:, 'weekday'] =  pd.to_datetime(df['rdate']).dt.weekday
    df.loc[:, 'diff_days'] = df.groupby(group)['rdate'].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
    df.loc[:, 'is_not_daily'] = df.apply(is_not_daily, axis=1)
    return df[df['is_not_daily'] == True]
    
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
# Functions - Show Calendar for Summary View
# =============================================================================

# Output week of the month based on the date
def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom/7.0))


# Return a dataframe of month, weeks, days in the given year
def year_cal(input_year):
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    df = pd.DataFrame({'month': pd.DatetimeIndex(dates).month_name(),
                       'weekday': pd.DatetimeIndex(dates).weekday,
                       'day': pd.DatetimeIndex(dates).day.astype(int),
                       'date': dates})

    
    week = df['date'].apply(week_of_month)
    df.insert(loc=1, column='week', value=week)
    return df

# Return bad dates based on which multiselect tables was chosen
def find_selected_bad_dates(tables):
    res_date = []
    for table in tables:
        if table == 'BMC Monthly':
            res_date.append(res_bmc_monthly['rdate'])
        elif table == 'Portfolio Holding':
            res_date.append(res_portholding['rdate'])
        elif table == 'Portfolio Return':
            res_date.append(res_portreturn['rdate'])
        elif table == 'BM Prices':
            res_date.append(res_bmprices['rdate'])
        elif table == 'Universe Snapshot':
            res_date.append(res_univsnapshot['rdate'])
        elif table == 'Div LTM':
            res_date.append(res_div_ltm['date'])

    if not res_date:
        st.warning("Please select a table to view.")
    else:
        res_date = pd.concat(res_date)
        res_date = res_date.unique()

    return res_date

# Show a dataframe with bad dates highlighted for the given month
def show_month_df(df, holiday_df, month):
    df = df[df['month'] == month]
    # df = [df[df['weekday'].isin(pd.Series(data=[0, 1, 2, 3, 4]))]]
    df = df.pivot(index='week', columns='weekday', values='day')
    dayOfWeek={0:'M', 1:'T', 2:'W', 3:'Th', 4:'F', 5:'S', 6:'Su'}
    df.columns = [df.columns.map(dayOfWeek)]
    df = df.fillna("")
    df = df.drop(['S', 'Su'], axis = 1)
    
    res_date_df = pd.DataFrame({
                    'month': pd.DatetimeIndex(res_date).month_name(),
                    'day': pd.DatetimeIndex(res_date).day,
                    'date': res_date})

    holiday_df = holiday_df[holiday_df['month'] == month]
    res_date_df = res_date_df[res_date_df['month'] == month]
    res_day = res_date_df['day']
    st.dataframe(df.style.apply(highlight_bad_day, args=[res_day, holiday_df['day']], axis=1).set_precision(0))


# Helper to highlight dates with bad data quality
def highlight_bad_day(days, res_day, holiday_days):
    # st.write(dates)
    yellow = 'background-color: orange'
    white = 'background-color: white'
    grey = 'background-color: darkgrey'

    colors = []
    for day in days:
        if day in res_day.values:
            colors.append(yellow)
        elif day in holiday_days.values:
            colors.append(grey)
        else: 
            colors.append(white)
    return colors

# Show 3 monthly calendars in a row
def show_months(holiday_df, m1, m2, m3):
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.header(m1)
        show_month_df(df, holiday_df, m1)
    with col2:
        st.header(m2)
        show_month_df(df,holiday_df, m2)
    with col3:
        st.header(m3)
        show_month_df(df,holiday_df, m3)



success_msg = 'No error found.'
error_msg = 'Error rows'


# =============================================================================
# Code
# =============================================================================

# Set page layout
st.set_page_config(layout="wide")
st.title('Data Quality Checker')
# st.text('Select a table to view details.')

# Load data
portholding = load_data(query_portholding)
bmprices = load_data(query_bmprices)
portreturn = load_data(query_portreturn)
bmc_monthly = load_data(query_bmc_monthly)
div_ltm = load_data(query_div_ltm)
univsnapshot = load_data(query_univsnapshot)

holiday = load_data(query_holiday)
# df.loc[:, 'num_stock'] = df['rdate'].groupby(df['rdate']).transform('count')

query_adjpricet = """SELECT [fsym_id]
  FROM [FSTest].[dbo].[AdjustedPriceTickers]"""
adjpricet = load_data(query_adjpricet)
# Differences in rdate between consec rows



res_bmc_monthly = find_null(bmc_monthly, 'fsym_id')
res_portholding = find_null(portholding, 'secid')
res_bmprices = find_daily(bmprices, 'bm_id')
res_portreturn = find_daily(portreturn, 'pid')
### Univ snapshot can't see the reason for which is which?
res_univsnapshot = find_univsnapshot(univsnapshot)
res_univ_notin_id = not_in_adjpricest(univsnapshot)
res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
res_div_ltm = find_null(div_ltm, 'date')
 
   
checkbox_sum = 'Summary'
checkbox_date = 'View by Date'
sum_view = st.checkbox(checkbox_sum)
date_view = st.checkbox(checkbox_date)

## monthly
### univsnap, bmc monthly, div_ltm
if not sum_view and not date_view:
    st.warning('Please select a view.')
else:
    if sum_view:
        st.header(checkbox_sum)
        # st.write(type(input_date))
        # input_year = st.number_input(
    #     'Select a  year', min_value = min_year, max_value=max_year,
    #     value=datetime.now().year, format='%d')
        input_year = st.number_input(
            'Select a  year', max_value=datetime.now().year,
            value=datetime.now().year, format='%d')

        res_portholding = res_portholding[res_portholding['rdate'].dt.year == input_year]
        res_bmprices = res_bmprices[res_bmprices['rdate'].dt.year == input_year]
        res_portreturn = res_portreturn[res_portreturn['rdate'].dt.year == input_year]
        res_bmc_monthly = res_bmc_monthly[res_bmc_monthly['rdate'].dt.year == input_year]
        res_div_ltm = res_div_ltm[res_div_ltm['date'].dt.year == input_year]
        res_univsnapshot = res_univsnapshot[res_univsnapshot['rdate'].dt.year == input_year]
        
        df = year_cal(input_year)
    
        lst_tables = ['BMC Monthly', 'Portfolio Holding', 'Portfolio Return', 'BM Prices', 'Universe Snapshot', 'Div LTM']
        selected = st.multiselect('Choose tables to view', lst_tables, ['Portfolio Holding'])
    
        res_date = find_selected_bad_dates(selected)            
        # res_date_df = pd.DataFrame({
        #                'month': pd.DatetimeIndex(res_date).month_name(),
        #                'day': pd.DatetimeIndex(res_date).day,
        #                'date': res_date})
        
        is_us_holiday = st.sidebar.checkbox('Show US Holiday')
        is_cad_holiday = st.sidebar.checkbox('Show Canadian Holiday')
        
        holiday_df = holiday[holiday['holiday_date'].dt.year == input_year]
        if not is_cad_holiday and not is_us_holiday:
            holiday_df = pd.DataFrame([], columns = ['fref_exchange_code', 'month', 'day'])
        else:
            holiday_df.loc[:, 'month'] = holiday_df['holiday_date'].dt.month_name()
            holiday_df.loc[:, 'day'] = holiday['holiday_date'].dt.day
            if is_us_holiday and not is_cad_holiday:
                holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'NYS']
            elif is_cad_holiday and not is_us_holiday:
                holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'TSE']
      
        # holiday_df.iloc[:, 'holiday_date'] = pd.to_datetime(holiday['holiday_date'], format='%Y-%m-%d')
        # st.write(type(holiday_df['holiday_date'].dt.month_name()))
        # st.write(type(holiday_df))
    
    
        
        show_months(holiday_df, 'January', 'February', 'March')
        show_months(holiday_df, 'April', 'May', 'June')
        show_months(holiday_df, 'July', 'August', 'September')
        show_months(holiday_df, 'October', 'November', 'December')
    
    
        # col1, col2, col3 = st.beta_columns(3)
        # with col1:
     
        # with col2:
    
        # with col3:
            # with col3:
    
            
        # st.header('Choose a table to view full result')
        # if st.button('BMC Monthly'):
        #     show_res(res_bmc_monthly)
        # if st.button('Portfolio Holding'):
        #     show_res(res_portholding)
        # if st.button('Portfolio Return'):
        #     show_res(res_portreturn)
        # if st.button('BM Price'):
        #     show_res(res_bmprices)
        # if st.button('Universe Snapshot'):
        #     show_res(res_univsnapshot)
        # if st.button('Div LTM'):
        #     show_res(res_div_ltm)
    
    
    if date_view:
        st.header('View by Date')
        input_date = st.date_input("Choose a date")
        
                # res_portholding.loc[:, 'rdate'] = pd.to_datetime(res_portholding['rdate'], format ='%Y-%m-%d')
        # res_portholding.loc[:, 'rdate'] = res_portholding['rdate'].dt.strftime('%Y-%m-%d')
    
        res_portholding = res_portholding[res_portholding['rdate'].dt.date == input_date]

        res_bmprices = res_bmprices[res_bmprices['rdate'].dt.date == input_date]
        res_portreturn = res_portreturn[res_portreturn['rdate'].dt.date == input_date]
    
        input_date_m = pd.Period(input_date,freq='M').end_time.date()
        res_bmc_monthly = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
                                          (res_bmc_monthly['rdate'].dt.month == input_date.month)]
        res_div_ltm = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
                                  (res_div_ltm['date'].dt.month == input_date.month)]
        res_univsnapshot = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
                                  (res_univsnapshot['rdate'].dt.month == input_date.month)]
        
    
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

