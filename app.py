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

#
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
    st.error(error_msg_prob_rows)
    st.write(res_portholding_daily)   

# Find errors of every table based on the input year and input table 
def find_res_tables(selected, input_year):
    if 'Portfolio Holding' in selected:
        res_portholding = portholding[portholding['rdate'].dt.year == input_year]
        res_portholding['rdate'] = res_portholding['rdate'].dt.date
    
        res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
        res_portholding = find_null(res_portholding, 'secid')
        # res_portholding = res_portholding.merge(res_portholding_is_daily.rename('rdate'), how='outer',on='rdate')
    else:
        res_portholding = pd.DataFrame([], columns = portholding.columns)
        res_portholding_is_daily = pd.Series([])

    if 'BM Prices' in selected:
        res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
        res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
        res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate']).to_series()
        
    else:
        res_bmprices = pd.Series([])

    if 'Portfolio Return' in selected:
        res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
        res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
        res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate']).to_series()
    else:
        res_portreturn = pd.Series([])
    if 'BMC Monthly' in selected:
        res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
        res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
        res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
        # res_bmc_monthly = res_bmc_monthly.merge(res_bmc_monthly_is_monthly.rename('rdate'), how='outer',on='rdate')
    else:
        res_bmc_monthly = pd.DataFrame([], columns = bmc_monthly.columns)
        res_bmc_monthly_is_monthly = pd.Series([])

    if 'Universe Snapshot' in selected:
        res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
        res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
        res_univsnapshot = find_univsnapshot(res_univsnapshot)
        res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
        res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
        # res_univsnapshot = res_univsnapshot.merge(res_univsnapshot_is_monthly.rename('rdate'), how='outer',on='rdate')
    else:
        res_univsnapshot = pd.DataFrame([], columns = univsnapshot.columns)
        res_univsnapshot_is_monthly = pd.Series([])
        res_univ_notin_id = pd.DataFrame([], columns = univsnapshot.columns)

    if 'Div LTM' in selected:
        res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
        res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
        res_div_ltm = find_null(res_div_ltm, 'date')
    else:
        res_div_ltm = pd.DataFrame([], columns = div_ltm.columns)
        res_div_ltm_is_monthly = pd.Series([])
    
    return res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn, res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly


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
    today = datetime.today()
    if edate > today:
        edate = today
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

def get_res_day(res_date, month):
    res_date_df = pd.DataFrame({
                'month': pd.DatetimeIndex(res_date).month_name(),
                'day': pd.DatetimeIndex(res_date).day,
                'date': res_date})
    res_date_df = res_date_df[res_date_df['month'] == month]
    return res_date_df['day']
    
    
# Show a dataframe with bad dates highlighted for the given month
def show_month_df(df, holiday_df, month):
    df = df[df['month'] == month]
    df = df.pivot(index='week', columns='weekday', values='day')
    dayOfWeek={0:'M', 1:'T', 2:'W', 3:'Th', 4:'F', 5:'S', 6:'Su'}
    df.columns = [df.columns.map(dayOfWeek)]
    
    df = df.fillna("")
    df = df.drop(['S', 'Su'], axis = 1)
    
    res_bmprices_day = get_res_day(res_bmprices, month)
    res_portreturn_day = get_res_day(res_portreturn, month)

    res_bmc_monthly_day = get_res_day(res_bmc_monthly['rdate'], month)
    res_portholding_day = get_res_day(res_portholding['rdate'], month)
    res_univsnapshot_day = get_res_day(res_univsnapshot['rdate'], month)
    res_div_ltm_day = get_res_day(res_div_ltm['date'], month)
        
    res_portholding_is_daily_day = get_res_day(res_portholding_is_daily, month)
    res_bmc_monthly_is_monthly_day = get_res_day(res_bmc_monthly_is_monthly, month)
    res_div_ltm_is_monthly_day = get_res_day(res_div_ltm_is_monthly, month)
    res_univsnapshot_is_monthly_day = get_res_day(res_univsnapshot_is_monthly, month)
    res_univ_notin_id_day = get_res_day(res_univ_notin_id['rdate'], month)
    holiday_df = holiday_df[holiday_df['month'] == month]
    checkToday = (month == datetime.today().strftime("%B"))
    st.dataframe(df.style.apply(highlight_bad_day, 
                                args=[checkToday, holiday_df['day'],
                                      res_bmprices_day, res_portreturn_day,
                                      res_bmc_monthly_day, res_portholding_day,
                                      res_univsnapshot_day, res_div_ltm_day,
                                      res_portholding_is_daily_day, res_bmc_monthly_is_monthly_day,
                                      res_div_ltm_is_monthly_day, res_univsnapshot_is_monthly_day,
                                      res_univ_notin_id_day], axis=1).set_precision(0))


# Helper to highlight dates with bad data quality
def highlight_bad_day(days, checkToday, holiday_days, res_bmprices_day, res_portreturn_day, 
                      res_bmc_monthly_day, res_portholding_day,
                      res_univsnapshot_day, res_div_ltm_day,
                        res_portholding_is_daily_day, res_bmc_monthly_is_monthly_day,
                        res_div_ltm_is_monthly_day, res_univsnapshot_is_monthly_day,
                        res_univ_notin_id_day):
    # st.write(dates)
    res_bmprices_color = 'background-color: orange'
    res_portreturn_color = 'background-color: yellow'
    res_bmc_monthly_color = 'background-color: purple'
    res_bmc_monthly_is_monthly_color = 'background-color: blueviolet'
    res_portholding_color = 'background-color: green'
    res_portholding_is_daily_color = 'background-color: lightgreen'
    res_univsnapshot_color = 'background-color: pink'
    res_univsnapshot_is_monthly_color = 'background-color: red'
    res_univ_notin_id_color = 'background-color: darkred'
    res_div_ltm_color = 'background-color: blue'
    res_div_ltm_is_monthly_color = 'background-color: lightblue'
    background_color = 'background-color: white'
    holiday_color = 'background-color: darkgrey'
    today_color = 'background-color: Aquamarine'
    if checkToday:
        today_day = datetime.today().date().day
    else:
        today_day = None        

    colors = []
    for day in days:

        if day in holiday_days.values:
            colors.append(holiday_color)
        elif day == today_day:
            colors.append(today_color)
        elif day in res_bmprices_day.values:
            colors.append(res_bmprices_color)
        elif day in res_portreturn_day.values:
            colors.append(res_portreturn_color)
        elif day in res_bmc_monthly_day.values:
            colors.append(res_bmc_monthly_color)
        elif day in res_portholding_day.values:
            colors.append(res_portholding_color)
        elif day in res_univsnapshot_day.values:
            colors.append(res_univsnapshot_color)
        elif day in res_div_ltm_day.values:
            colors.append(res_div_ltm_color)
        elif day in res_bmc_monthly_is_monthly_day.values:
            colors.append(res_bmc_monthly_is_monthly_color)
        elif day in res_portholding_is_daily_day.values:
            colors.append(res_portholding_is_daily_color)
        elif day in res_univsnapshot_is_monthly_day.values:
            colors.append(res_univsnapshot_is_monthly_color)
        elif day in res_univ_notin_id_day.values:
            colors.append(res_univ_notin_id_color)
        elif day in res_div_ltm_is_monthly_day.values:
            colors.append(res_div_ltm_is_monthly_color) 
        else: 
            colors.append(background_color)
    return colors

# Show 3 monthly calendars in a row
def show_months(df, holiday_df, m1, m2, m3):
    col1, col2, col3 = st.beta_columns(3)
    cur_month = datetime.today().month
    months = dict(January=1, February=2, March=3, April=4, May=5,
                  June=6, July=7, August=8, September=9, October=10,
                  November=11, December=12)
    with col1:
        if months[m1] <= cur_month:
            st.header(m1)
            show_month_df(df, holiday_df, m1)
    with col2:
        if months[m2] <= cur_month:
            st.header(m2)
            show_month_df(df,holiday_df, m2)
    with col3:
        if months[m3] <= cur_month:
            st.header(m3)
            show_month_df(df,holiday_df, m3)

# Highlight explanation dataframe in different color
def highlight_color(row):
    table = row.Table
    reason = row.Reason
    res_bmprices_color = 'background-color: orange'
    res_portreturn_color = 'background-color: yellow'
    res_bmc_monthly_color = 'background-color: purple'
    res_bmc_monthly_is_monthly_color = 'background-color: blueviolet'
    res_portholding_color = 'background-color: green'
    res_portholding_is_daily_color = 'background-color: lightgreen'
    res_univsnapshot_color = 'background-color: pink'
    res_univsnapshot_is_monthly_color = 'background-color: red'
    res_univ_notin_id_color = 'background-color: darkred'
    res_div_ltm_color = 'background-color: blue'
    res_div_ltm_is_monthly_color = 'background-color: lightblue'
    holiday_color = 'background-color: darkgrey'
    today_color = 'background-color: Aquamarine'

    if table == bmprices_label:
        return 2 * [res_bmprices_color]
    if table == portreturn_label:
        return 2 * [res_portreturn_color]
    if table == portholding_label:
        if reason == not_updated_daily_reason:
            return 2 * [res_portholding_is_daily_color]
        if reason == prob_row_reason:
            return 2 *  [res_portholding_color]
    if table == bmc_monthly_label:
        if reason == not_updated_monthly_reason:
            return 2 * [res_bmc_monthly_is_monthly_color]
        if reason == prob_row_reason:
            return 2 *  [res_bmc_monthly_color]
    if table == univsnapshot_label:
        if reason == not_updated_monthly_reason:
            return 2 * [res_univsnapshot_is_monthly_color]
        if reason == large_port_diff_reason:
            return 2 * [res_univ_notin_id_color]
        if reason == prob_row_reason:
            return 2 *  [res_univsnapshot_color]
    if table == div_ltm_label:
        if reason == not_updated_monthly_reason:
            return 2 * [res_div_ltm_is_monthly_color]
        if reason == prob_row_reason:
            return 2 *  [res_div_ltm_color]
    if table == holiday_label:
        return 2 * [holiday_color]
    if table == today_label:
        return 2 * [today_color]
    
success_msg = 'No problematic rows found.'
error_msg_prob_rows = 'Found problematic rows.'


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

st.sidebar.subheader('Other views:')
date_view = st.sidebar.checkbox(checkbox_date)

st.sidebar.subheader('Holidays:')
is_us_holiday = st.sidebar.checkbox('Show US Holiday', value=True)
is_cad_holiday = st.sidebar.checkbox('Show Canadian Holiday')
    
## monthly
### univsnap, bmc monthly, div_ltm

portholding_label = 'Portfolio Holding'
portreturn_label = 'Portfolio Return'
bmprices_label = 'BM Prices'
bmc_monthly_label = 'BMC Monthly'
univsnapshot_label = 'Universe Snapshot'
div_ltm_label = 'Div LTM'
holiday_label = 'Holiday'
today_label = 'Today'
lst_tables = ['BM Prices', 'Portfolio Return', 'Portfolio Holding', 'Portfolio Holding',
              'BMC Monthly', 'BMC Monthly', 'Universe Snapshot','Universe Snapshot',
              'Universe Snapshot', div_ltm_label,div_ltm_label, holiday_label, today_label]
not_updated_daily_reason ='Not Updated Daily'
not_updated_monthly_reason ='Not Updated Monthly'
prob_row_reason ='Problematic Rows'
large_port_diff_reason = 'Large monthly portfolio count differences'
color_df = pd.DataFrame({'Table': lst_tables,
    'Reason': [not_updated_daily_reason,not_updated_daily_reason, 
              prob_row_reason, not_updated_daily_reason,
              prob_row_reason, not_updated_monthly_reason,
               prob_row_reason,large_port_diff_reason , not_updated_monthly_reason,
               prob_row_reason, not_updated_monthly_reason, holiday_label, today_label]})


# Sum view
st.header(checkbox_sum)
input_year = st.number_input(
    'Select a  year', min_value = 1990, max_value=datetime.now().year,
    value=datetime.now().year, format='%d')

holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)

year_cal = year_cal(input_year)
lst_tables = ['BMC Monthly', 'Portfolio Holding', 'Portfolio Return', 'BM Prices', 'Universe Snapshot', 'Div LTM']
selected = st.multiselect('Choose tables to view', lst_tables, ['Portfolio Holding'])

show_table_color_ref = st.checkbox('Show color references')
if show_table_color_ref:
    st.dataframe(color_df.style.apply(highlight_color, axis=1))
st.info('There is a 1 day data updatelag for some tables.')

res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn,res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly = find_res_tables(selected, input_year)

show_months(year_cal, holiday_df, 'January', 'February', 'March')
show_months(year_cal, holiday_df, 'April', 'May', 'June')
show_months(year_cal, holiday_df, 'July', 'August', 'September')
show_months(year_cal, holiday_df, 'October', 'November', 'December')

    
if date_view:

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
    res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate'])
    res_portholding = find_null(res_portholding, 'secid')



    ### Univ snapshot can't see the reason for which is which?        
    res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    res_univsnapshot = find_univsnapshot(res_univsnapshot)
    res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
    res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    res_div_ltm = find_null(res_div_ltm, 'date')
    
    # res_portreturn_daily
    # res_bmprices_daily
    res_bmprices = res_bmprices.date
    res_bmprices = pd.Series(res_bmprices)
    res_portreturn = res_portreturn.date
    res_portreturn = pd.Series(res_portreturn)
    res_portholding_is_daily = res_portholding_is_daily.date
    res_portholding_is_daily = pd.Series(res_portholding_is_daily)
    
    res_bmprices_daily = res_bmprices[res_bmprices == input_date]   
    res_portreturn_daily = res_portreturn[res_portreturn == input_date]
    res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]

    res_portholding_is_daily_daily = res_portholding_is_daily[res_portholding_is_daily == input_date]


    res_bmc_monthly_daily = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
                                      (res_bmc_monthly['rdate'].dt.month == input_date.month)]
    res_bmc_monthly_is_monthly_daily = res_bmc_monthly_is_monthly[(res_bmc_monthly_is_monthly.dt.year == input_date.year) &
                                      (res_bmc_monthly_is_monthly.dt.month == input_date.month)]
    res_div_ltm_daily = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
                              (res_div_ltm['date'].dt.month == input_date.month)]
    res_div_ltm_is_monthly_daily = res_div_ltm_is_monthly[(res_div_ltm_is_monthly.dt.year == input_date.year) &
                              (res_div_ltm_is_monthly.dt.month == input_date.month)]
    res_univsnapshot_daily = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
                              (res_univsnapshot['rdate'].dt.month == input_date.month)]
    res_univsnapshot_is_monthly_daily = res_univsnapshot_is_monthly[(res_univsnapshot_is_monthly.dt.year == input_date.year) &
                              (res_univsnapshot_is_monthly.dt.month == input_date.month)]
    res_univ_notin_id_daily = res_univ_notin_id[(res_univ_notin_id['rdate'].dt.year == input_date.year) &
                              (res_univ_notin_id['rdate'].dt.month == input_date.month)]
  

    if st.button('BM Price'):
        if res_bmprices_daily.empty:
            st.success(success_msg)
        else:
            st.error('No data found on ' + str(input_date))
    if st.button('Portfolio Return'):
        if res_portreturn_daily.empty:
            st.success(success_msg)
        else:
            st.error('No data found on ' + str(input_date))
    if st.button('Portfolio Holding'):
        if res_portholding_is_daily_daily.empty and res_portholding_daily.empty:
            st.success(success_msg)
        if res_portholding_is_daily_daily.empty and not res_portholding_daily.empty:
            show_res(res_portholding_daily)
        if not res_portholding_is_daily_daily.empty and res_portholding_daily.empty:
            st.error('No data found on ' + str(input_date))           
        if not res_portholding_is_daily_daily.empty and not res_portholding_daily.empty:
            st.error('No data found on ' + str(input_date))
            show_res(res_portholding_daily)
    if st.button('BMC Monthly'):
        if res_bmc_monthly_is_monthly_daily.empty and res_bmc_monthly_daily.empty:
            st.success(success_msg)
        if res_bmc_monthly_is_monthly_daily.empty and not res_bmc_monthly_daily.empty:
            show_res(res_bmc_monthly_daily)
        if not res_bmc_monthly_is_monthly_daily.empty and res_bmc_monthly_daily.empty:
            st.error('No data found on ' + str(input_date))           
        if not res_bmc_monthly_is_monthly_daily.empty and not res_bmc_monthly_daily.empty:
            st.error('No data found on ' + str(input_date))
            show_res(res_bmc_monthly_daily)
    if st.button('Universe Snapshot'):
        if (res_univsnapshot_daily.empty and res_univ_notin_id_daily.empty) and res_univsnapshot_is_monthly_daily.empty:
            st.success(success_msg)
        if res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
                                                        (not res_univsnapshot_daily.empty)):
            show_res(res_univsnapshot_daily)
            show_res(res_univ_notin_id_daily)
        if res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
                                                        (res_univsnapshot_daily.empty)):
            show_res(res_univ_notin_id_daily)
        if res_univsnapshot_is_monthly_daily.empty and ((res_univ_notin_id_daily.empty) and
                                                        (not res_univsnapshot_daily.empty)):
            show_res(res_univsnapshot_daily)    
        if not res_univsnapshot_is_monthly_daily.empty and (( res_univ_notin_id_daily.empty) and
                                                        (res_univsnapshot_daily.empty)):
            st.error('No data found on ' + str(input_date))           
        if not res_univsnapshot_is_monthly_daily.empty and ((res_univ_notin_id_daily.empty) and
                                                        (not res_univsnapshot_daily.empty)):
            st.error('No data found on ' + str(input_date))
            show_res(res_univsnapshot_daily)
        if not res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
                                                         res_univsnapshot_daily.empty):
            st.error('No data found on ' + str(input_date))
            show_res(res_univ_notin_id_daily)
        if not res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
                                                        (not res_univsnapshot_daily.empty)):
            st.error('No data found on ' + str(input_date))
            show_res(res_univsnapshot_daily)
            show_res(res_univ_notin_id_daily)
    if st.button('Div LTM'):
        if res_div_ltm_is_monthly_daily.empty and res_univsnapshot_is_monthly_daily.empty:
            st.success(success_msg)
        if res_div_ltm_is_monthly_daily.empty and not res_div_ltm_daily.empty:
            show_res(res_div_ltm_daily)
        if not res_div_ltm_is_monthly_daily.empty and res_div_ltm_daily.empty:
            st.error('No data found on ' + str(input_date))           
        if not res_div_ltm_is_monthly_daily.empty and not res_div_ltm_daily.empty:
            st.error('No data found on ' + str(input_date))
            show_res(res_div_ltm_daily)

# if sum_view and not date_view:
    
#     st.header(checkbox_sum)
#     input_year = st.number_input(
#         'Select a  year', min_value = 1990, max_value=datetime.now().year,
#         value=datetime.now().year, format='%d')
    
#     holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)

#     df = year_cal(input_year)
#     lst_tables = ['BMC Monthly', 'Portfolio Holding', 'Portfolio Return', 'BM Prices', 'Universe Snapshot', 'Div LTM']
#     selected = st.multiselect('Choose tables to view', lst_tables, ['Portfolio Holding'])
    
#     show_table_color_ref = st.checkbox('Show color references')
#     if show_table_color_ref:
#         st.dataframe(color_df.style.apply(highlight_color, axis=1))

#     # Updated daily
#     res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn,res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly = find_res_tables(selected, input_year)

#     ### Univ snapshot can't see the reason for which is which?        
        

#     show_months(df, holiday_df, 'January', 'February', 'March')
#     show_months(df, holiday_df, 'April', 'May', 'June')
#     show_months(df, holiday_df, 'July', 'August', 'September')
#     show_months(df, holiday_df, 'October', 'November', 'December')

# if not sum_view and date_view:
#     st.header('View by Date')
    
#     input_date = st.date_input("Choose a date")
#     input_year = input_date.year
    
#     holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)
    
    
#     res_portholding = portholding[portholding['rdate'].dt.year == input_year]
#     res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
#     res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
#     res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
#     res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
#     res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
    
#     # Updated daily
#     res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
#     res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
#     res_portholding['rdate'] = res_portholding['rdate'].dt.date

#     res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate'])
#     res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate'])
#     res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate'])
#     res_portholding = find_null(res_portholding, 'secid')
#     # res_portholding = res_portholding.merge(res_portholding_is_daily.rename('rdate'), how='outer',on='rdate')

      



#     ### Univ snapshot can't see the reason for which is which?        
#     res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
#     res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
#     res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
#     res_univsnapshot = find_univsnapshot(res_univsnapshot)
#     res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
#     res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
#     res_div_ltm = find_null(res_div_ltm, 'date')
    
    
#     res_bmprices = res_bmprices.date
#     res_bmprices = pd.Series(res_bmprices)
#     res_portreturn = res_portreturn.date
#     res_portreturn = pd.Series(res_portreturn)
#     res_portholding_is_daily = res_portholding_is_daily.date
#     res_portholding_is_daily = pd.Series(res_portholding_is_daily)
    
    

#     res_bmprices_daily = res_bmprices[res_bmprices == input_date]    
#     res_portreturn_daily = res_portreturn[res_portreturn == input_date]
#     res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]

#     res_portholding_is_daily_daily = res_portholding_is_daily[res_portholding_is_daily == input_date]


#     res_bmc_monthly_daily = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
#                                       (res_bmc_monthly['rdate'].dt.month == input_date.month)]
#     res_bmc_monthly_is_monthly_daily = res_bmc_monthly_is_monthly[(res_bmc_monthly_is_monthly.dt.year == input_date.year) &
#                                       (res_bmc_monthly_is_monthly.dt.month == input_date.month)]
#     res_div_ltm_daily = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
#                               (res_div_ltm['date'].dt.month == input_date.month)]
#     res_div_ltm_is_monthly_daily = res_div_ltm_is_monthly[(res_div_ltm_is_monthly.dt.year == input_date.year) &
#                               (res_div_ltm_is_monthly.dt.month == input_date.month)]
#     res_univsnapshot_daily = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
#                               (res_univsnapshot['rdate'].dt.month == input_date.month)]
#     res_univsnapshot_is_monthly_daily = res_univsnapshot_is_monthly[(res_univsnapshot_is_monthly.dt.year == input_date.year) &
#                               (res_univsnapshot_is_monthly.dt.month == input_date.month)]
#     res_univ_notin_id_daily = res_univ_notin_id[(res_univ_notin_id['rdate'].dt.year == input_date.year) &
#                               (res_univ_notin_id['rdate'].dt.month == input_date.month)]
  

#     if st.button('BM Price'):
#         if res_bmprices_daily.empty:
#             st.success(success_msg)
#         else:
#             st.error('No data found on ' + str(input_date))
#     if st.button('Portfolio Return'):
#         if res_portreturn_daily.empty:
#             st.success(success_msg)
#         else:
#             st.error('No data found on ' + str(input_date))
#     if st.button('Portfolio Holding'):
#         if res_portholding_is_daily_daily.empty and res_portholding_daily.empty:
#             st.success(success_msg)
#         if res_portholding_is_daily_daily.empty and not res_portholding_daily.empty:
#             show_res(res_portholding_daily)
#         if not res_portholding_is_daily_daily.empty and res_portholding_daily.empty:
#             st.error('No data found on ' + str(input_date))           
#         if not res_portholding_is_daily_daily.empty and not res_portholding_daily.empty:
#             st.error('No data found on ' + str(input_date))
#             show_res(res_portholding_daily)
#     if st.button('BMC Monthly'):
#         if res_bmc_monthly_is_monthly_daily.empty and res_bmc_monthly_daily.empty:
#             st.success(success_msg)
#         if res_bmc_monthly_is_monthly_daily.empty and not res_bmc_monthly_daily.empty:
#             show_res(res_bmc_monthly_daily)
#         if not res_bmc_monthly_is_monthly_daily.empty and res_bmc_monthly_daily.empty:
#             st.error('No data found on ' + str(input_date))           
#         if not res_bmc_monthly_is_monthly_daily.empty and not res_bmc_monthly_daily.empty:
#             st.error('No data found on ' + str(input_date))
#             show_res(res_bmc_monthly_daily)
#     if st.button('Universe Snapshot'):
#         if (res_univsnapshot_daily.empty and res_univ_notin_id_daily.empty) and res_univsnapshot_is_monthly_daily.empty:
#             st.success(success_msg)
#         if res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
#                                                         (not res_univsnapshot_daily.empty)):
#             show_res(res_univsnapshot_daily)
#             show_res(res_univ_notin_id_daily)
#         if res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
#                                                         (res_univsnapshot_daily.empty)):
#             show_res(res_univ_notin_id_daily)
#         if res_univsnapshot_is_monthly_daily.empty and ((res_univ_notin_id_daily.empty) and
#                                                         (not res_univsnapshot_daily.empty)):
#             show_res(res_univsnapshot_daily)    
#         if not res_univsnapshot_is_monthly_daily.empty and (( res_univ_notin_id_daily.empty) and
#                                                         (res_univsnapshot_daily.empty)):
#             st.error('No data found on ' + str(input_date))           
#         if not res_univsnapshot_is_monthly_daily.empty and ((res_univ_notin_id_daily.empty) and
#                                                         (not res_univsnapshot_daily.empty)):
#             st.error('No data found on ' + str(input_date))
#             show_res(res_univsnapshot_daily)
#         if not res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
#                                                          res_univsnapshot_daily.empty):
#             st.error('No data found on ' + str(input_date))
#             show_res(res_univ_notin_id_daily)
#         if not res_univsnapshot_is_monthly_daily.empty and ((not res_univ_notin_id_daily.empty) and
#                                                         (not res_univsnapshot_daily.empty)):
#             st.error('No data found on ' + str(input_date))
#             show_res(res_univsnapshot_daily)
#             show_res(res_univ_notin_id_daily)
#     if st.button('Div LTM'):
#         if res_div_ltm_is_monthly_daily.empty and res_univsnapshot_is_monthly_daily.empty:
#             st.success(success_msg)
#         if res_div_ltm_is_monthly_daily.empty and not res_div_ltm_daily.empty:
#             show_res(res_div_ltm_daily)
#         if not res_div_ltm_is_monthly_daily.empty and res_div_ltm_daily.empty:
#             st.error('No data found on ' + str(input_date))           
#         if not res_div_ltm_is_monthly_daily.empty and not res_div_ltm_daily.empty:
#             st.error('No data found on ' + str(input_date))
#             show_res(res_div_ltm_daily)

