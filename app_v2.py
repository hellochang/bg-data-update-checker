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

# holiday_date = holiday['holiday_date'].dt.date
# Return True if the dates differ more or less than 1. Helper for find_daily(df, group)
def is_not_daily(row, holiday_date):
    diff_days = row.diff_days
    # yesterday = (row.rdate - timedelta(1)).date()
    yesterday = row.rdate - timedelta(1)
    last_friday = row.rdate - timedelta(3)
    # st.write(type(holiday_date)
    # Check for weekends and remove Mondays
    # st.write(diff_days)
    # st.write(diff_days)
    # if diff_days == timedelta(1):
    #     return False
    # elif (row.weekday == 0) and (diff_days == timedelta(3)):
    #     return False
    # elif (yesterday in holiday_date.values) and (diff_days == timedelta(2)):
    #     # st.write(yesterday)
    #     # st.write(holiday['holiday_date'])
    #     return False
    # elif (row.weekday == 0 and diff_days == timedelta(4)) and last_friday in holiday_date.values:
    #     return False
    
    # For weekdays
    if diff_days == 1:
        return False
    # elif ((row.rdate.month == 1) and (row.rdate.day == 1)):
    #     return False
    elif (row.weekday == 0) and diff_days == 3:
        return False
    elif (yesterday in holiday_date.values):
        if (diff_days == 2) or (row.weekday == 1 and diff_days == 4):
        # st.write(yesterday)
        # st.write(holiday['holiday_date'])
            return False
    elif (row.weekday == 0 and diff_days == 4) and last_friday in holiday_date.values:
        return False
    else:
        return True

# Differences in rdate between consec rows
@st.cache
def find_daily(df, group, holiday_date):
    df = df.copy()
    df['weekday'] =  pd.to_datetime(df['rdate']).dt.weekday
    df['diff_days'] = df.groupby(group)['rdate'].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
    # df['diff1'] = df.groupby(group)['rdate'].shift(1)
    # df['diff_days'] = df['rdate'] - df['diff1']
    # df['diff_days'].fillna(0)
    df.loc[df.groupby(group)['diff_days'].head(1).index, 'diff_days'] = 1

    df['is_not_daily'] = df.apply(is_not_daily, args=[holiday_date], axis=1)
    # st.write(df[df['rdate']=='2021-01-01'])
    return df[df['is_not_daily'] == True]

def is_in_table_dates(week_date, table_dates):
    return week_date in table_dates.values

    
@st.cache
def find_univsnapshot(df):
    tol = 300
    df = df.copy()
    df['monthly_company_count'] = df['univ_id'].groupby(df['rdate'].dt.month).transform('count')
    df['diff_monthly'] = df['monthly_company_count'].diff()

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
    """ Returns the week of the month for the specified date."""
    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom/7.0))


# Return a dataframe of month, weeks, days in the given year
def year_cal(input_year):
    """Return a dataframe of yearly calendar."""

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

def weekday_dates(input_year, holiday_date):
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    dates = dates[dates.weekday < 5]
    dates = dates[~dates.isin(holiday_date)]
    return dates
    
# Return bad dates based on which multiselect tables was chosen
def find_selected_bad_dates(tables):
    res_date = []
    for table in tables:
        if table == 'BMC Monthly':
            res_date.append(res_bmc_monthly['rdate'])
        elif table == 'Portfolio Holding':
            res_date.append(res_portholding['rdate'])
        elif table == 'Portfolio Return':
            res_date.append(res_portreturn.to_series())
        elif table == 'BM Prices':
            res_date.append(res_bmprices.to_series())
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
    highlight = 'background-color: orange'
    white = 'background-color: white'
    grey = 'background-color: darkgrey'

    colors = []
    for day in days:
        if day in holiday_days.values:
            colors.append(grey)
        elif day in res_day.values:
            colors.append(highlight)
        else: 
            colors.append(white)
    return colors

# Show 3 monthly calendars in a row
def show_months(df, holiday_df, m1, m2, m3):
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



success_msg = 'No problematic rows found.'
error_msg = 'Found problematic rows.'


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


query_adjpricet = """SELECT [fsym_id]
  FROM [FSTest].[dbo].[AdjustedPriceTickers]"""
adjpricet = load_data(query_adjpricet)
# Differences in rdate between consec rows

def check_daily(input_year, holiday_date, res_df_date):
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    business_dates = dates[dates.weekday < 5]
    business_dates = dates[~dates.isin(holiday_date)]
    return business_dates[~business_dates.isin(res_df_date)]

def check_monthly(input_year, res_bmc_monthly, col):
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    monthly_dates_uniq = [datetime(input_year, 1, 1), datetime(input_year, 2, 1), datetime(input_year, 3, 1),
    datetime(input_year, 4, 1), datetime(input_year, 5, 1), datetime(input_year, 6, 1),
    datetime(input_year, 7, 1), datetime(input_year, 8, 1), datetime(input_year, 9, 1),
    datetime(input_year, 10, 1), datetime(input_year, 11, 1), datetime(input_year, 12, 1)]
    monthly_dates_uniq = pd.Series(monthly_dates_uniq)
    monthly_dates = pd.date_range(start=sdate, end=edate)
    
    monthly_dates = pd.Series(monthly_dates)

    res_bmc_monthly = res_bmc_monthly.copy()
    monthly_dates_not_in_res = pd.Series(monthly_dates_uniq[~monthly_dates_uniq.dt.month.isin(res_bmc_monthly[col].dt.month)])
    return monthly_dates[monthly_dates.dt.month.isin(monthly_dates_not_in_res.dt.month)]
    

def get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday):

    holiday_df = holiday[holiday['holiday_date'].dt.year == input_year]
    if not is_cad_holiday and not is_us_holiday:
        holiday_df = pd.DataFrame([], columns = ['fref_exchange_code', 'holiday_date', 'month', 'day'])
    else:
        holiday_df.loc[:, 'month'] = holiday_df['holiday_date'].dt.month_name()
        holiday_df.loc[:, 'day'] = holiday['holiday_date'].dt.day
        if is_us_holiday and not is_cad_holiday:
            holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'NYS']
        elif is_cad_holiday and not is_us_holiday:
            holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'TSE']
    
    if not holiday_df.empty:
        holiday_date = holiday_df['holiday_date'].dt.date
    else:
        holiday_date = pd.Series([])
    
    return holiday_df, holiday_date

checkbox_sum = 'Summary'
checkbox_date = 'View by Date'
sum_view = st.checkbox(checkbox_sum, value=True)
date_view = st.checkbox(checkbox_date)

is_us_holiday = st.sidebar.checkbox('Show US Holiday', value=True)
is_cad_holiday = st.sidebar.checkbox('Show Canadian Holiday')
    
## monthly
### univsnap, bmc monthly, div_ltm
if not sum_view and not date_view:
    st.warning('Please select a view.')



if sum_view and date_view:
    st.header(checkbox_sum)
    input_year = st.number_input(
        'Select a  year', min_value = 1990, max_value=datetime.now().year,
        value=datetime.now().year, format='%d')

    res_portholding = portholding[portholding['rdate'].dt.year == input_year]
    res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
    res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
    res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
    res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
    res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
    


        
    holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)
    # Updated daily
    res_portholding['rdate'] = res_portholding['rdate'].dt.date
    res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
    res_portreturn['rdate'] = res_portreturn['rdate'].dt.date

    res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate'])
    res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate'])
    res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
    res_portholding = find_null(res_portholding, 'secid')
    res_portholding = res_portholding.merge(res_portholding_is_daily.rename('rdate'), how='outer',on='rdate')

      

    ### Univ snapshot can't see the reason for which is which?        
    res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    res_bmc_monthly = res_bmc_monthly.merge(res_bmc_monthly_is_monthly.rename('rdate'), how='outer',on='rdate')

    res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    res_univsnapshot = find_univsnapshot(res_univsnapshot)
    res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
    res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
    res_univsnapshot = res_univsnapshot.merge(res_univsnapshot_is_monthly.rename('rdate'), how='outer',on='rdate')
    
    
    res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    res_div_ltm = find_null(res_div_ltm, 'date')
    res_div_ltm = res_div_ltm.merge(res_div_ltm_is_monthly.rename('date'), how='outer',on='date')



    df = year_cal(input_year)
    lst_tables = ['BMC Monthly', 'Portfolio Holding', 'Portfolio Return', 'BM Prices', 'Universe Snapshot', 'Div LTM']
    selected = st.multiselect('Choose tables to view', lst_tables, ['Portfolio Holding'])
    res_date = find_selected_bad_dates(selected)            

    show_months(df, holiday_df, 'January', 'February', 'March')
    show_months(df, holiday_df, 'April', 'May', 'June')
    show_months(df, holiday_df, 'July', 'August', 'September')
    show_months(df, holiday_df, 'October', 'November', 'December')
    
    
    
    st.header('View by Date')
    
    input_date = st.date_input("Choose a date")
    input_year = input_date.year
    
    
    
    
    res_bmprices = res_bmprices.date
    res_bmprices = pd.Series(res_bmprices)
    res_portreturn = res_portreturn.date
    res_portreturn = pd.Series(res_portreturn)

    res_bmprices_daily = res_bmprices[res_bmprices == input_date]    
    # st.write(type(res_bmprices_daily))
    
    res_portreturn_daily = res_portreturn[res_portreturn == input_date]
    res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]

    input_date_m = pd.Period(input_date,freq='M').end_time.date()
    res_bmc_monthly_daily = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
                                      (res_bmc_monthly['rdate'].dt.month == input_date.month)]
    res_div_ltm_daily = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
                              (res_div_ltm['date'].dt.month == input_date.month)]
    res_univsnapshot_daily = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
                              (res_univsnapshot['rdate'].dt.month == input_date.month)]
    


    if st.button('Portfolio Holding'):
        show_res(res_portholding_daily)
    if st.button('Portfolio Return'):
        show_res(res_portreturn_daily)
    if st.button('BM Price'):
        show_res(res_bmprices_daily)
    if st.button('BMC Monthly'):
        show_res(res_bmc_monthly_daily)
    if st.button('Universe Snapshot'):
        show_res(res_univsnapshot_daily)
    if st.button('Div LTM'):
        show_res(res_div_ltm_daily)


if sum_view and not date_view:
    st.header(checkbox_sum)
    input_year = st.number_input(
        'Select a  year', min_value = 1990, max_value=datetime.now().year,
        value=datetime.now().year, format='%d')

    res_portholding = portholding[portholding['rdate'].dt.year == input_year]
    res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
    res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
    res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
    res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
    res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
    


        
    holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)
    # Updated daily
    res_portholding['rdate'] = res_portholding['rdate'].dt.date
    res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
    res_portreturn['rdate'] = res_portreturn['rdate'].dt.date

    res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate'])
    res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate'])
    res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
    res_portholding = find_null(res_portholding, 'secid')
    res_portholding = res_portholding.merge(res_portholding_is_daily.rename('rdate'), how='outer',on='rdate')

      

    ### Univ snapshot can't see the reason for which is which?        
    res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    res_bmc_monthly = res_bmc_monthly.merge(res_bmc_monthly_is_monthly.rename('rdate'), how='outer',on='rdate')

    res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    res_univsnapshot = find_univsnapshot(res_univsnapshot)
    res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
    res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
    res_univsnapshot = res_univsnapshot.merge(res_univsnapshot_is_monthly.rename('rdate'), how='outer',on='rdate')
    
    
    res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    res_div_ltm = find_null(res_div_ltm, 'date')
    res_div_ltm = res_div_ltm.merge(res_div_ltm_is_monthly.rename('date'), how='outer',on='date')



    df = year_cal(input_year)
    lst_tables = ['BMC Monthly', 'Portfolio Holding', 'Portfolio Return', 'BM Prices', 'Universe Snapshot', 'Div LTM']
    selected = st.multiselect('Choose tables to view', lst_tables, ['Portfolio Holding'])
    res_date = find_selected_bad_dates(selected)            

    show_months(df, holiday_df, 'January', 'February', 'March')
    show_months(df, holiday_df, 'April', 'May', 'June')
    show_months(df, holiday_df, 'July', 'August', 'September')
    show_months(df, holiday_df, 'October', 'November', 'December')

if not sum_view and date_view:
    st.header('View by Date')
    
    input_date = st.date_input("Choose a date")
    input_year = input_date.year
    
    
    
    holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)
    
    
    
    res_portholding = portholding[portholding['rdate'].dt.year == input_year]
    res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
    res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
    res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
    res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
    res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
    
    # Updated daily
    res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
    res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
    res_portholding['rdate'] = res_portholding['rdate'].dt.date

    res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate'])
    res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate'])
    res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
    res_portholding = find_null(res_portholding, 'secid')
    res_portholding = res_portholding.merge(res_portholding_is_daily.rename('rdate'), how='outer',on='rdate')

      



    ### Univ snapshot can't see the reason for which is which?        
    res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    res_bmc_monthly = res_bmc_monthly.merge(res_bmc_monthly_is_monthly.rename('rdate'), how='outer',on='rdate')

    res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    res_univsnapshot = find_univsnapshot(res_univsnapshot)
    res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
    res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
    res_univsnapshot = res_univsnapshot.merge(res_univsnapshot_is_monthly.rename('rdate'), how='outer',on='rdate')
    
    
    res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    res_div_ltm = find_null(res_div_ltm, 'date')
    res_div_ltm = res_div_ltm.merge(res_div_ltm_is_monthly.rename('date'), how='outer',on='date')
    
    
    
    
    
    
    
    res_bmprices = res_bmprices.date
    res_bmprices = pd.Series(res_bmprices)
    res_portreturn = res_portreturn.date
    res_portreturn = pd.Series(res_portreturn)

    res_bmprices_daily = res_bmprices[res_bmprices == input_date]    
    # res_portreturn_daily = res_portreturn[res_portreturn == input_date]
    res_portreturn_daily = pd.DataFrame({'rdate': res_portreturn[res_portreturn == input_date]}, 
                                        columns=portreturn.columns)
    
    res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]

    input_date_m = pd.Period(input_date,freq='M').end_time.date()
    res_bmc_monthly_daily = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
                                      (res_bmc_monthly['rdate'].dt.month == input_date.month)]
    res_div_ltm_daily = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
                              (res_div_ltm['date'].dt.month == input_date.month)]
    res_univsnapshot_daily = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
                              (res_univsnapshot['rdate'].dt.month == input_date.month)]
    


    if st.button('Portfolio Holding'):
        show_res(res_portholding_daily)
    if st.button('Portfolio Return'):
        show_res(res_portreturn_daily)
    if st.button('BM Price'):
        show_res(res_bmprices_daily)
    if st.button('BMC Monthly'):
        show_res(res_bmc_monthly_daily)
    if st.button('Universe Snapshot'):
        show_res(res_univsnapshot_daily)
    if st.button('Div LTM'):
        show_res(res_div_ltm_daily)
