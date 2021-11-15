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
# Variables
# =============================================================================
# Queries
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
query_adjpricet = """SELECT [fsym_id]
  FROM [FSTest].[dbo].[AdjustedPriceTickers]"""
query_holiday = """SELECT [fref_exchange_code]
      ,[holiday_date]
      ,[holiday_name]
  FROM [FSTest].[ref_v2].[ref_calendar_holidays]
WHERE fref_exchange_code IN ('NYS', 'TSE')"""

# Widget Labels
checkbox_sum = 'Summary'
checkbox_date = 'View by Date'
portholding_label = 'Portfolio Holding'
portreturn_label = 'Portfolio Return'
bmprices_label = 'BM Prices'
bmc_monthly_label = 'BMC Monthly'
univsnapshot_label = 'Universe Snapshot'
div_ltm_label = 'Div LTM'
holiday_label = 'Holiday'
today_label = 'Today'

# Messages
not_updated_daily_reason ='Not Updated Daily'
not_updated_monthly_reason ='Not Updated Monthly'
prob_row_reason ='Problematic Rows'
large_port_diff_reason = 'Large monthly portfolio count differences'
success_msg = 'No problematic rows found.'
error_msg_prob_rows = 'Found problematic rows.'

# Setup
lst_tables = [bmc_monthly_label, portholding_label, portreturn_label, bmprices_label, univsnapshot_label, div_ltm_label]
lst_tables_colors = [bmprices_label, portreturn_label, portholding_label, portholding_label,
              bmc_monthly_label, bmc_monthly_label, univsnapshot_label,univsnapshot_label,
              univsnapshot_label, div_ltm_label,div_ltm_label, holiday_label, today_label]
color_df = pd.DataFrame({'Table': lst_tables_colors,
    'Reason': [not_updated_daily_reason,not_updated_daily_reason, 
              prob_row_reason, not_updated_daily_reason,
              prob_row_reason, not_updated_monthly_reason,
               prob_row_reason,large_port_diff_reason , not_updated_monthly_reason,
               prob_row_reason, not_updated_monthly_reason, holiday_label, today_label]})

# Colors for highlights and reference table
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
background_color = 'background-color: white'

# =============================================================================
# Functions - Import data
# =============================================================================

# Perform query and fetch data from SQL server.
@st.cache(allow_output_mutation=True, ttl=60*60)
def load_data(query):
    """Load data from SQL database based on query and returns a dataframe"""
    data = DataImporter(verbose=False)
    return data.load_data(query)

# =============================================================================
# Functions - Check Data Quality
# =============================================================================

@st.cache
def find_null(df, col):
    """Filters dataframe for rows that contain null values in input column"""
    return df[df[col].isnull()]

#
@st.cache
def find_univsnapshot(df):
    """Filters dataframe for rows with monthly company count larger than tolerance"""
    tol = 300
    df = df.copy()
    df['monthly_company_count'] = df['univ_id'].groupby(df['rdate'].dt.month).transform('count')
    df['diff_monthly'] = df['monthly_company_count'].diff()

    return df[df['diff_monthly'] > tol]

@st.cache
def not_in_adjpricest(df):
    """Filters dataframe for rows not also in Adj Price table"""
    adjpricet_fsym_id = adjpricet['fsym_id'].unique()  
    return df[~df['fsym_id'].isin(adjpricet_fsym_id)]

@st.cache
def check_daily(input_year, holiday_date, res_df_date):
    """Returns business dates not included in input df"""
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    business_dates = dates[dates.weekday < 5]
    business_dates = dates[~dates.isin(holiday_date)]
    return business_dates[~business_dates.isin(res_df_date)]

@st.cache
def check_monthly(input_year, res_df_date, col):
    sdate = datetime(input_year, 1, 1)  
    edate = datetime(input_year, 12, 31)
    monthly_dates_uniq = [datetime(input_year, 1, 1), datetime(input_year, 2, 1), datetime(input_year, 3, 1),
    datetime(input_year, 4, 1), datetime(input_year, 5, 1), datetime(input_year, 6, 1),
    datetime(input_year, 7, 1), datetime(input_year, 8, 1), datetime(input_year, 9, 1),
    datetime(input_year, 10, 1), datetime(input_year, 11, 1), datetime(input_year, 12, 1)]
    monthly_dates_uniq = pd.Series(monthly_dates_uniq)
    monthly_dates = pd.date_range(start=sdate, end=edate)
    
    monthly_dates = pd.Series(monthly_dates)

    res_df_date = res_df_date.copy()
    monthly_dates_not_in_res = pd.Series(monthly_dates_uniq[~monthly_dates_uniq.dt.month.isin(res_df_date[col].dt.month)])
    return monthly_dates[monthly_dates.dt.month.isin(monthly_dates_not_in_res.dt.month)]


def find_res_tables(selected, input_year):
    """Find errors of every table based on the input year and input table """
    if portholding_label in selected:
        res_portholding = portholding[portholding['rdate'].dt.year == input_year]
        res_portholding['rdate'] = res_portholding['rdate'].dt.date
        res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
        res_portholding = find_null(res_portholding, 'secid')
    else:
        res_portholding = pd.DataFrame([], columns = portholding.columns)
        res_portholding_is_daily = pd.Series([])

    if bmprices_label in selected:
        res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
        res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
        res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate']).to_series()
        
    else:
        res_bmprices = pd.Series([])

    if portreturn_label in selected:
        res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
        res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
        res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate']).to_series()
    else:
        res_portreturn = pd.Series([])
    if bmc_monthly_label in selected:
        res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
        res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
        res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    else:
        res_bmc_monthly = pd.DataFrame([], columns = bmc_monthly.columns)
        res_bmc_monthly_is_monthly = pd.Series([])

    if univsnapshot_label in selected:
        res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
        res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
        res_univsnapshot = find_univsnapshot(res_univsnapshot)
        res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
        res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
    else:
        res_univsnapshot = pd.DataFrame([], columns = univsnapshot.columns)
        res_univsnapshot_is_monthly = pd.Series([])
        res_univ_notin_id = pd.DataFrame([], columns = univsnapshot.columns)

    if div_ltm_label in selected:
        res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
        res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
        res_div_ltm = find_null(res_div_ltm, 'date')
    else:
        res_div_ltm = pd.DataFrame([], columns = div_ltm.columns)
        res_div_ltm_is_monthly = pd.Series([])
    
    return res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn, res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly

def show_res_df(header, res_daily_df, res_df=None, res_df_2=None):
    """Dispaly results for daily view based on how many things each table is checking"""
    st.subheader(header)
    if (res_daily_df is not None and res_df is None) and res_df_2 is None:
        if res_daily_df.empty:
            st.success(success_msg)
        if not res_daily_df.empty:
            st.error('No data found on ' + str(input_date))
        
    if (res_daily_df is not None and res_df is not None) and res_df_2 is None:
        if res_daily_df.empty and res_df.empty:
            st.success(success_msg)
        if res_daily_df.empty and not res_df.empty:
            st.error(error_msg_prob_rows)
            st.write(res_df)  
        if not res_daily_df.empty and res_df.empty:
            st.error('No data found on ' + str(input_date))           
        if not res_daily_df.empty and not res_df.empty:
            st.error('No data found on ' + str(input_date))
            st.error(error_msg_prob_rows)
            st.write(res_df) 
            
    if (res_daily_df is not None and res_df is not None) and res_df_2 is not None:
        if (res_daily_df.empty and res_df.empty) and res_univsnapshot_is_monthly_daily.empty:
                st.success(success_msg)
        if res_daily_df.empty and ((not res_df.empty) and not res_df_2.empty):
            st.error(error_msg_prob_rows)
            st.write(res_univsnapshot_daily)  
            st.error(error_msg_prob_rows)
            st.write(res_univ_notin_id_daily)  
    
        if res_daily_df.empty and ((not res_df.empty) and
                                                        (res_df_2.empty)):
            st.error(error_msg_prob_rows)
            st.write(res_univ_notin_id_daily)  
        if res_daily_df.empty and ((res_df.empty) and
                                                        (not res_df_2.empty)):
            st.error(error_msg_prob_rows)
            st.write(res_univsnapshot_daily) 
        if not res_daily_df.empty and (( res_df.empty) and
                                                        (res_df_2.empty)):
            st.error('No data found on ' + str(input_date))           
        if not res_daily_df.empty and ((res_df.empty) and
                                                        (not res_df_2.empty)):
            st.error('No data found on ' + str(input_date))
            st.error(error_msg_prob_rows)
            st.write(res_univsnapshot_daily)
        if not res_daily_df.empty and ((not res_df.empty) and
                                                        res_df_2.empty):
            st.error('No data found on ' + str(input_date))
            st.error(error_msg_prob_rows)
            st.write(res_univ_notin_id_daily)  
        if not res_daily_df.empty and ((not res_df.empty) and
                                                        (not res_df_2.empty)):
            st.error('No data found on ' + str(input_date))
            st.error(error_msg_prob_rows)
            st.write(res_univsnapshot_daily)  
            st.error(error_msg_prob_rows)
            st.write(res_univ_notin_id_daily)

def show_res_daily_view(selected):
    """Display result for Daily View"""
    for table in selected:
        if table == bmprices_label:
            show_res_df(bmprices_label, res_bmprices_daily)
        elif table == portreturn_label:
            show_res_df(portreturn_label, res_portreturn_daily)
        elif table == portholding_label:
            show_res_df(portholding_label, res_portholding_is_daily_daily, res_portholding_daily)
        elif table == bmc_monthly_label:
            show_res_df(bmc_monthly_label, res_bmc_monthly_is_monthly_daily, res_bmc_monthly_daily)
        elif table == univsnapshot_label:
            show_res_df(univsnapshot_label, res_univsnapshot_is_monthly_daily, res_univ_notin_id_daily, res_univsnapshot_daily)
        elif table == div_ltm_label:
            show_res_df(div_ltm_label, res_div_ltm_is_monthly_daily, res_div_ltm_daily)

@st.cache
def get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday):
    """Return holidays in the given year based on which holiday calendar is selected"""
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
  

def get_res_day(res_date, month):
    res_date_df = pd.DataFrame({
                'month': pd.DatetimeIndex(res_date).month_name(),
                'day': pd.DatetimeIndex(res_date).day,
                'date': res_date})
    res_date_df = res_date_df[res_date_df['month'] == month]
    return res_date_df['day']
    
# month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# year_df = pd.DataFrame({'year': range(1990, datetime.now.year)}, columns=month_names)

# Show a dataframe with bad dates highlighted for the given month
def show_month_df(input_year, df, holiday_df, month):
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
    
    today = datetime.today()
    months = dict(January=1, February=2, March=3, April=4, May=5,
                  June=6, July=7, August=8, September=9, October=10,
                  November=11, December=12)
    isToday = ((month == today.strftime("%B")) and input_year == today.year)
    isMonthLaterThanToday = months[month] > today.month
    st.dataframe(df.style.apply(highlight_bad_day, 
                                args=[isToday, isMonthLaterThanToday, holiday_df['day'],
                                      res_bmprices_day, res_portreturn_day,
                                      res_bmc_monthly_day, res_portholding_day,
                                      res_univsnapshot_day, res_div_ltm_day,
                                      res_portholding_is_daily_day, res_bmc_monthly_is_monthly_day,
                                      res_div_ltm_is_monthly_day, res_univsnapshot_is_monthly_day,
                                      res_univ_notin_id_day], axis=1).set_precision(0))


def show_months(input_year, holiday_df, m1, m2, m3):
    """ Show monthly calendars for three months in a row"""
    col1, col2, col3 = st.beta_columns(3)
    year_calendar = year_cal(input_year)
    with col1:
        st.header(m1)
        show_month_df(input_year, year_calendar, holiday_df, m1)
    with col2:
        st.header(m2)
        show_month_df(input_year, year_calendar,holiday_df, m2)
    with col3:
        st.header(m3)
        show_month_df(input_year, year_calendar,holiday_df, m3)


def highlight_bad_day(days, isToday, isMonthLaterThanToday, holiday_days, res_bmprices_day, res_portreturn_day, 
                      res_bmc_monthly_day, res_portholding_day,
                      res_univsnapshot_day, res_div_ltm_day,
                        res_portholding_is_daily_day, res_bmc_monthly_is_monthly_day,
                        res_div_ltm_is_monthly_day, res_univsnapshot_is_monthly_day,
                        res_univ_notin_id_day):
    """Helper function to highlight dates with bad data quality in the calendar df"""

    if isToday:
        today_day = datetime.today().date().day
    else:
        today_day = None        

    colors = []
    for day in days:
        if day != "":
            if isMonthLaterThanToday:
                colors.append(background_color)
            elif today_day is None or int(day) <= today_day:
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
            else:
                colors.append(background_color)
        else:
            colors.append(background_color)
    return colors


def highlight_color(row):
    """Highlight color reference dataframe in different color"""
    table = row.Table
    reason = row.Reason

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

# =============================================================================
# Code
# =============================================================================

# Set page layout
st.set_page_config(layout="wide")
st.title('Data Quality Checker')

# Load data
portholding = load_data(query_portholding)
bmprices = load_data(query_bmprices)
portreturn = load_data(query_portreturn)
bmc_monthly = load_data(query_bmc_monthly)
div_ltm = load_data(query_div_ltm)
univsnapshot = load_data(query_univsnapshot)
holiday = load_data(query_holiday)
adjpricet = load_data(query_adjpricet)


st.sidebar.subheader('Other views:')
date_view = st.sidebar.checkbox(checkbox_date)

st.sidebar.subheader('Holidays:')
is_us_holiday = st.sidebar.checkbox('Show US Holiday', value=True)
is_cad_holiday = st.sidebar.checkbox('Show Canadian Holiday')


# Sum view
st.header(checkbox_sum)
input_year = st.number_input(
    'Select a year', min_value = 1990, max_value=datetime.now().year,
    value=datetime.now().year, format='%d')

holiday_df, holiday_date = get_holiday(input_year, holiday, is_us_holiday, is_cad_holiday)

selected = st.multiselect('Select a table to view details.', lst_tables, [portholding_label])

show_table_color_ref = st.checkbox('Show color references')
if show_table_color_ref:
    st.dataframe(color_df.style.apply(highlight_color, axis=1))
st.info('There is a 1 day data update lag for some tables.')

res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn,res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly = find_res_tables(selected, input_year)

show_months(input_year, holiday_df, 'January', 'February', 'March')
show_months(input_year, holiday_df, 'April', 'May', 'June')
show_months(input_year, holiday_df, 'July', 'August', 'September')
show_months(input_year, holiday_df, 'October', 'November', 'December')

    
if date_view:

    st.header('View by Date')
    
    input_date = st.date_input("Choose a date", max_value=datetime.today())
    
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


    # Updated monthly     
    res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    res_univsnapshot = find_univsnapshot(res_univsnapshot)
    res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
    res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    res_div_ltm = find_null(res_div_ltm, 'date')
    
    # res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn,res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly = find_res_tables(selected, input_year)
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

    show_res_daily_view(selected)
    