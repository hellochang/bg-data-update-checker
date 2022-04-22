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
from typing import Optional, List, Tuple, Dict

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
# =============================================================================
# Variables
# =============================================================================
# Queries
query_bmc_monthly = """SELECT * FROM development.dbo.bmc_monthly 
WHERE bm_id IN ('sp500', 'sptsx')"""
query_univsnapshot = """SELECT *
FROM fstest.dbo.univsnapshot
WHERE univ_id IN ('CANADA', 'US')"""
query_portholding = """SELECT * FROM development.dbo.portholding
WHERE company NOT IN ('Placeholder', 'Cash-INVEST-USD', 'Cash-INVEST-CAD')"""
query_div_ltm = """SELECT div.fsym_id, exdate, date, div_type, div_freq
                    FROM fstest.dbo.bg_div AS div
                    LEFT JOIN fstest.dbo.bg_div_ltm AS ltm
                    ON div.exdate = ltm.date
                    AND div.fsym_id = ltm.fsym_id
                    WHERE  div_type ='regular'
                    AND dummy_payment = 0
                """
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


# Messages
reason_dict = {'not_updated_daily': 'Not Updated Daily',
               'not_updated_monthly': 'Not Updated Monthly',
               'prob_row': 'Problematic Rows',
               'large_port_diff': 'Large monthly portfolio count differences',
               }

msg = {'success': 'No problematic rows found.',
       'error_prob_rows': 'Found problematic rows.',
       'multiselect_table': 'Select a table to view details.',
       'update_lag': 'There is a 1 day data update lag for some tables.',
       'year_selection': 'Select a year'
       }


# Widget Labels
header = {
    'holiday': 'Holidays:',
    'other_view': 'Other views:',
    'sum_view': 'Summary'
}
checkbox_label = {
    'date_view': 'View by Date',
    'us_holiday': 'Show US Holiday',
    'cad_holiday': 'Show Canadian Holiday',
    'color_ref': 'Show color references',
}

table_label = {
    'portholding': 'Portfolio Holding',
    'portreturn': 'Portfolio Return',
    'bmprices': 'BM Prices',
    'bmc_monthly': 'BMC Monthly',
    'univsnapshot': 'Universe Snapshot',
    'div_ltm': 'Div LTM',
    'holiday': 'Holiday',
    'today': 'Today'
}


# Setup
lst_tables = [table_label['bmc_monthly'],
              table_label['portholding'],
              table_label['portreturn'],
              table_label['bmprices'],
              table_label['univsnapshot'],
              table_label['div_ltm']]
lst_tables_colors = [table_label['bmprices'],
                     table_label['portreturn'],
                     table_label['portholding'],
                     table_label['portholding'],
                     table_label['bmc_monthly'],
                     table_label['bmc_monthly'],
                     table_label['univsnapshot'],
                     table_label['univsnapshot'],
                     table_label['univsnapshot'],
                     table_label['div_ltm'],
                     table_label['div_ltm'],
                     table_label['holiday'],
                     table_label['today']]
color_df = pd.DataFrame({'Table': lst_tables_colors,
                        'Reason': [reason_dict['not_updated_daily'],
                                   reason_dict['not_updated_daily'],
                                   reason_dict['prob_row'],
                                   reason_dict['not_updated_daily'],
                                   reason_dict['prob_row'],
                                   reason_dict['not_updated_monthly'],
                                   reason_dict['prob_row'],
                                   reason_dict['large_port_diff'],
                                   reason_dict['not_updated_monthly'],
                                   reason_dict['prob_row'],
                                   reason_dict['not_updated_monthly'],
                                   table_label['holiday'],
                                   table_label['today']]})

# Colors for highlights and reference table
colors = {'bmprices': 'background-color: orange',
          'portreturn': 'background-color: yellow',
          'bmc_monthly': 'background-color: purple',
          'bmc_monthly_is_monthly': 'background-color: blueviolet',
          'portholding': 'background-color: green',
          'portholding_is_daily': 'background-color: lightgreen',
          'univsnapshot': 'background-color: pink',
          'univsnapshot_is_monthly': 'background-color: red',
          'univ_notin_id': 'background-color: darkred',
          'div_ltm': 'background-color: blue',
          'div_ltm_is_monthly': 'background-color: lightblue',
          'holiday': 'background-color: darkgrey',
          'today': 'background-color: aquamarine',
          'background': 'background-color: white'
          }


# =============================================================================
# Functions - Import data
# =============================================================================

# Perform query and fetch data from SQL server.
@st.cache(allow_output_mutation=True, ttl=60*60)
def load_data(query: str) -> pd.DataFrame:
    """
    Load data from SQL database based on query and returns a dataframe

    Parameters
    ----------
    query : str
        SQL query.

    Returns
    -------
    pd.DataFrame
        A dataframe of data loaded from MS SQL DB.

    """
    data = DataImporter(verbose=False)
    return data.load_data(query)


# =============================================================================
# Functions - Check Data Quality
# =============================================================================

@st.cache
def find_univsnapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters dataframe for rows with monthly company count larger than tolerance

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of data that we need to check, loaded from DB

    Returns
    -------
    pd.DataFrame
        Rows with monthly company count larger than tolerance

    """
    tol = 300
    df = df.copy()
    df = df.groupby('rdate')['univ_id'].value_counts()
    df = df.reset_index(name='monthly_company_count')
    df['diff_monthly'] = df.groupby('monthly_company_count')['univ_id'].diff()
    return df[df['diff_monthly'] > tol]


@st.cache
def check_daily(input_year: int, holiday_date: pd.Series,
                df_date: pd.Series) -> pd.DataFrame:
    """
    Returns business dates not included in input df

    Parameters
    ----------
    input_year : int
        The year that we need to check (that the user selected).
    holiday_date : pd.Series
        Holiday dates.
    df_date : pd.Series
        Dataframe of data that we need to check, loaded from DB

    Returns
    -------
    res : pd.DataFrame
         Business dates not present in input df that we need to check.

    """
    sdate = datetime(input_year, 1, 1)
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    weekday_dates = dates[dates.weekday < 5]
    business_dates = weekday_dates[~weekday_dates.isin(holiday_date)]
    res = pd.DataFrame(
        {'rdate': business_dates[~business_dates.isin(df_date)].date})
    return res


@st.cache
def check_monthly(input_year: int,
                  df_date: pd.Series) -> pd.DataFrame:
    """
    Returns dates not included in input Series

    Parameters
    ----------
    input_year : int
        The year that we need to check.
    df_date : pd.Series
        Dataframe of data that we need to check, loaded from DB

    Returns
    -------
    res : pd.DataFrame
        Dates in month that are not present in input_df.

    """
    sdate = datetime(input_year, 1, 1)
    edate = datetime(input_year, 12, 31)
    monthly_dates_uniq = pd.date_range(start=sdate, end=edate, freq='M')
    monthly_dates = pd.date_range(start=sdate, end=edate)
    monthly_dates_uniq = pd.Series(monthly_dates_uniq)
    monthly_dates = pd.Series(monthly_dates)

    df_date = df_date.copy()
    monthly_dates_not_in_res = pd.Series(monthly_dates_uniq[
        ~monthly_dates_uniq.dt.month.isin(df_date['rdate'].dt.month)])
    res = monthly_dates[monthly_dates.dt.month.isin(
        monthly_dates_not_in_res.dt.month)]
    res = res.dt.date.to_frame('rdate')
    return res


def get_result_tables(selected: List[str], input_year: str,
                      data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Find errors of every table based on the input year and input table


    Parameters
    ----------
    selected : List[str]
        Tables selected in the Streamlit multiselect widget.
    input_year : str
        Year we need to check.
    data : Dict[str, pd.DataFrame]
        Data loaded from DB.

    Returns
    -------
    result_dict : Dict[str, pd.DataFrame]
        Error entries in data after performing the checks.

    """
    if table_label['portholding'] in selected:
        portholding = data['portholding']
        res_portholding = portholding[portholding['rdate'].dt.year == input_year]
        res_portholding = res_portholding.copy()
        res_portholding['rdate'] = res_portholding['rdate'].dt.date
        res_portholding_is_daily = check_daily(input_year, holiday_date,
                                               res_portholding['rdate'])
        res_portholding = res_portholding[res_portholding['secid'].isnull()]
    else:
        res_portholding = pd.DataFrame([], columns=data['portholding'].columns)
        res_portholding_is_daily = pd.DataFrame([], columns=['rdate'])

    if table_label['bmprices'] in selected:
        bmprices = data['bmprices']
        res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
        res_bmprice = res_bmprices.copy()
        res_bmprice['rdate'] = res_bmprices['rdate'].dt.date
        res_bmprices = check_daily(input_year, holiday_date,
                                   res_bmprices['rdate'])
    else:
        res_bmprices = pd.DataFrame([], columns=['rdate'])

    if table_label['portreturn'] in selected:
        portreturn = data['portreturn']
        res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year]
        res_portreturn = res_portreturn.copy()
        res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
        res_portreturn = check_daily(input_year, holiday_date,
                                     res_portreturn['rdate'])
    else:
        res_portreturn = pd.DataFrame([], columns=['rdate'])

    if table_label['bmc_monthly'] in selected:
        bmc_monthly = data['bmc_monthly']
        res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
        res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly)
        res_bmc_monthly = res_bmc_monthly[res_bmc_monthly['fsym_id'].isnull()]
    else:
        res_bmc_monthly = pd.DataFrame([], columns=data['bmc_monthly'].columns)
        res_bmc_monthly_is_monthly = pd.DataFrame([], columns=['rdate'])

    if table_label['univsnapshot'] in selected:
        univsnapshot = data['univsnapshot']
        res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
        res_univsnapshot_is_monthly = check_monthly(
            input_year, res_univsnapshot)

        # Filters dataframe for rows not also in Adj Price table
        adjpricet_fsym_id = data['adjpricet']['fsym_id'].unique()
        res_univ_notin_id = res_univsnapshot[
            ~res_univsnapshot['fsym_id'].isin(adjpricet_fsym_id)]

        res_univsnapshot = find_univsnapshot(res_univsnapshot)
    else:
        res_univsnapshot = pd.DataFrame(
            [], columns=data['univsnapshot'].columns)
        res_univsnapshot_is_monthly = pd.DataFrame([], columns=['rdate'])
        res_univ_notin_id = pd.DataFrame(
            [], columns=data['univsnapshot'].columns)

    if table_label['div_ltm'] in selected:
        res_div_ltm = data['div_ltm'][data['div_ltm']
                                      ['rdate'].dt.year == input_year]
        res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm)
        res_div_ltm = res_div_ltm[res_div_ltm['rdate'].isnull()]
    else:
        res_div_ltm = pd.DataFrame([], columns=data['div_ltm'].columns)
        res_div_ltm_is_monthly = pd.DataFrame([], columns=['rdate'])

    result_dict = {'portholding': res_portholding,
                   'portholding_is_daily': res_portholding_is_daily,
                   'bmprices': res_bmprices,
                   'portreturn': res_portreturn,
                   'bmc_monthly': res_bmc_monthly,
                   'bmc_monthly_is_monthly': res_bmc_monthly_is_monthly,
                   'univsnapshot': res_univsnapshot,
                   'univsnapshot_is_monthly': res_univsnapshot_is_monthly,
                   'univ_notin_id': res_univ_notin_id,
                   'div_ltm': res_div_ltm,
                   'div_ltm_is_monthly': res_div_ltm_is_monthly}
    return result_dict


def get_result_daily(input_date: datetime.date,
                     result_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Get problematic entries for each table for a given date.

    Parameters
    ----------
    input_date : datetime.date
        Date that we need to check.
    result_dict : Dict[str, pd.DataFrame]
        Dictionary that contains problematic entries for each table

    Returns
    -------
    result_dict_daily : Dict[str, pd.DataFrame]
        Dictionary of problematic entries for each table for a given date.

    """
    tables = ['bmc_monthly', 'bmc_monthly_is_monthly', 'portholding',
              'univsnapshot', 'univsnapshot_is_monthly', 'univ_notin_id',
              'div_ltm', 'div_ltm_is_monthly']
    result_dict_daily = {tbl: result_dict[tbl][
        (pd.DatetimeIndex(
            result_dict[tbl]['rdate']).year == input_date.year) &
        (pd.DatetimeIndex(
            result_dict[tbl]['rdate']).month == input_date.month)]
        for tbl in tables}
    tables_daily = ['bmprices', 'portreturn', 'portholding_is_daily']
    result_dict_daily_2 = {
        tbl: result_dict[tbl][result_dict[tbl]['rdate'] == input_date]
        for tbl in tables_daily}
    result_dict_daily.update(result_dict_daily_2)

    return result_dict_daily


@st.cache
def get_holiday(input_year: int, holiday: pd.Series,
                is_us_holiday: bool,
                is_cad_holiday: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return holidays in the given year based on which holiday calendar is selected

    Parameters
    ----------
    input_year : int
        The year that we need to check.
    holiday : pd.Series
        The holidays loaded from SQL.
    is_us_holiday : bool
        If we need to check for US holidays.
    is_cad_holiday : bool
        If we need to check for Canadian holidays.

    Returns
    -------
    holiday_date : pd.Series
        Series of holiday dates for the given year for the given country.

    """
    holiday_df = holiday[holiday['holiday_date'].dt.year == input_year]
    if not is_cad_holiday and not is_us_holiday:
        holiday_date = pd.Series([])
        return holiday_date
    if is_us_holiday and not is_cad_holiday:
        holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'NYS']
    if is_cad_holiday and not is_us_holiday:
        holiday_df = holiday_df[holiday_df['fref_exchange_code'] == 'TSE']
    holiday_date = holiday_df['holiday_date'].dt.date

    return holiday_date


def get_result_sum_df(input_year: str,
                      holiday_date: pd.Series,
                      result_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get a summary df to show result in calendar view

    Parameters
    ----------
    input_year : str
        The year that the user selected.
    holiday_date : pd.Series
        Series contains the holidays this year.
    result_dict : Dict[str, pd.DataFrame]
        Dictionary that contains problematic entries for each table

    Returns
    -------
    res_table_df : pd.DataFrame
        Dataframe that contains the error code for each table and date in the
        given year.

    """
    sdate = datetime(input_year, 1, 1)
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    res_table_df = pd.DataFrame([], index=dates.date)

    for tbl in result_dict.keys():
        res_table_df[tbl] = dates.isin(result_dict[tbl]['rdate'])

    res_table_df['holiday'] = dates.isin(holiday_date)

    return res_table_df


# =============================================================================
# Functions - Show Calendar for Summary View
# =============================================================================

# Output week of the month based on the date
def week_of_month(dt: datetime) -> int:
    """
    Returns the week of the month for the specified date.

    Parameters
    ----------
    dt : datetime
        A date.

    Returns
    -------
    int
        The week of the month for the given date.

    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom/7.0))


# Return a dataframe of month, weeks, days in the given year
def year_cal(input_year: int) -> pd.DataFrame:
    """
    Return a dataframe of yearly calendar.

    Parameters
    ----------
    input_year : int
        The year that the user selected to view.

    Returns
    -------
    None.
    """
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


def show_months(input_year: str, res_table_df: pd.DataFrame, m1: str,
                m2: str, m3: str) -> None:
    """
    Show monthly calendars for three months in a row

    Parameters
    ----------
    input_year : str
        The year that the user selected to view.
    res_table_df : pd.DataFrame
        Dataframe contains the error code for each table.
    m1 : str
        First month we wanted to display.
    m2 : str
        Second month we wanted to display.
    m3 : str
        Third month we wanted to display.

    Returns
    -------
    None

    """
    col1, col2, col3 = st.beta_columns(3)
    year_calendar = year_cal(input_year)

    with col1:
        st.header(m1)
        show_month_df(input_year, year_calendar, res_table_df, m1)
    with col2:
        st.header(m2)
        show_month_df(input_year, year_calendar, res_table_df, m2)
    with col3:
        st.header(m3)
        show_month_df(input_year, year_calendar, res_table_df, m3)


def show_month_df(input_year: str, df: pd.DataFrame,
                  res_table_df: pd.DataFrame, month: str) -> None:
    """
    Show a dataframe with bad dates highlighted for the given month

    Parameters
    ----------
    input_year : str
        The year that the user selected to view.
    df : pd.DataFrame
        The dataframe of the given year.
    res_table_df : pd.DataFrame
        Dataframe contains the error code for each table.
    month : str
        The month we want to display.

    Returns
    -------
    None

    """
    df = df[df['month'] == month]
    df = df.pivot(index='week', columns='weekday', values='day')
    dayOfWeek = {0: 'M', 1: 'T', 2: 'W', 3: 'Th', 4: 'F', 5: 'S', 6: 'Su'}
    df.columns = [df.columns.map(dayOfWeek)]
    df = df.fillna("")
    df = df.drop(['S', 'Su'], axis=1, level=0)

    today = datetime.today()
    months = dict(January=1, February=2, March=3, April=4, May=5,
                  June=6, July=7, August=8, September=9, October=10,
                  November=11, December=12)
    isToday = ((month == today.strftime("%B")) and input_year == today.year)
    isMonthLaterThanToday = months[month] > today.month
    res_table_df = res_table_df[pd.DatetimeIndex(res_table_df.index)
                                  .month_name() == month]
    res_table_df.index = pd.DatetimeIndex(res_table_df.index).day
    st.dataframe(df.style.apply(highlight_bad_day,
                                args=[isToday, isMonthLaterThanToday,
                                      res_table_df], axis=1).set_precision(0))


def highlight_bad_day(days: pd.Series, isToday: bool, isMonthLaterThanToday: bool,
                      res_df: pd.DataFrame) -> List[str]:
    """
    Helper function to highlight dates with bad data quality in the calendar df

    Parameters
    ----------
    days : pd.Series
        The days in a week.
    isToday : bool
        Whether today is in the month that this function is highlighting.
    isMonthLaterThanToday : bool
        Whether the month that this function is highlighting is later than today.
    res_df : pd.DataFrame
        Dataframe contains the error code for each table.

    Returns
    -------
    List[str]
        List of colors to color a week in the monthly dataframe.

    """
    if isToday:
        today_day = datetime.today().date().day
    else:
        today_day = None

    res_colors = []
    for day in days:
        if day != "":
            if isMonthLaterThanToday:
                res_colors.append(colors['background'])
            elif today_day is None or int(day) <= today_day:
                if res_df.at[day, 'holiday']:
                    res_colors.append(colors['holiday'])
                elif day == today_day:
                    res_colors.append(colors['today'])
                elif res_df.at[day, 'bmprices']:
                    res_colors.append(colors['bmprices'])
                elif res_df.at[day, 'portreturn']:
                    res_colors.append(colors['portreturn'])
                elif res_df.at[day, 'portholding_is_daily']:
                    res_colors.append(colors['portholding_is_daily'])
                elif res_df.at[day, 'bmc_monthly']:
                    res_colors.append(colors['bmc_monthly'])
                elif res_df.at[day, 'portholding']:
                    res_colors.append(colors['portholding'])
                elif res_df.at[day, 'univsnapshot']:
                    res_colors.append(colors['univsnapshot'])
                elif res_df.at[day, 'div_ltm']:
                    res_colors.append(colors['div_ltm'])
                elif res_df.at[day, 'bmc_monthly_is_monthly']:
                    res_colors.append(colors['bmc_monthly_is_monthly'])
                elif res_df.at[day, 'univsnapshot_is_monthly']:
                    res_colors.append(colors['univsnapshot_is_monthly'])
                elif res_df.at[day, 'div_ltm_is_monthly']:
                    res_colors.append(colors['div_ltm_is_monthly'])
                elif res_df.at[day, 'univ_notin_id']:
                    res_colors.append(colors['univ_notin_id'])
                else:
                    res_colors.append(colors['background'])
            else:
                res_colors.append(colors['background'])
        else:
            res_colors.append(colors['background'])
    return res_colors


def highlight_color(row: pd.Series) -> List[str]:
    """
    Highlight color reference dataframe in different color

    Parameters
    ----------
    row : pd.Series
        A row in the color reference dataframe.

    Returns
    -------
    List[str]
        List of colors for the current row.

    """
    table = row.Table
    reason = row.Reason
    if table == table_label['bmprices']:
        return 2 * [colors['bmprices']]
    if table == table_label['portreturn']:
        return 2 * [colors['portreturn']]
    if table == table_label['portholding']:
        if reason == reason_dict['not_updated_daily']:
            return 2 * [colors['portholding_is_daily']]
        if reason == reason_dict['prob_row']:
            return 2 * [colors['portholding']]
    if table == table_label['bmc_monthly']:
        if reason == reason_dict['not_updated_monthly']:
            return 2 * [colors['bmc_monthly_is_monthly']]
        if reason == reason_dict['prob_row']:
            return 2 * [colors['bmc_monthly']]
    if table == table_label['univsnapshot']:
        if reason == reason_dict['not_updated_monthly']:
            return 2 * [colors['univsnapshot_is_monthly']]
        if reason == reason_dict['large_port_diff']:
            return 2 * [colors['univ_notin_id']]
        if reason == reason_dict['prob_row']:
            return 2 * [colors['univsnapshot']]
    if table == table_label['div_ltm']:
        if reason == reason_dict['not_updated_monthly']:
            return 2 * [colors['div_ltm_is_monthly']]
        if reason == reason_dict['prob_row']:
            return 2 * [colors['div_ltm']]
    if table == table_label['holiday']:
        return 2 * [colors['holiday']]
    if table == table_label['today']:
        return 2 * [colors['today']]


# =============================================================================
# Functions - Show Result for Daily View
# =============================================================================

def show_res_df(header: str, res_daily_df: pd.DataFrame,
                res_df: Optional[pd.DataFrame] = None,
                res_df_2: Optional[pd.DataFrame] = None) -> None:
    """
    Dispaly results for daily view based on items each table is checking

    Parameters
    ----------
    header : str
        The header displayed on the dashboard.
    res_daily_df : pd.DataFrame
        Problematic entries of a table for not updated daily or monthly.
    res_df : Optional[pd.DataFrame], optional
        Problematic entries of a table for another reason.
    res_df_2 : Optional[pd.DataFrame], optional
        Problematic entries of a table for the third reason, if needed.

    Returns
    -------
    None

    """
    st.subheader(header)

    error_flag = False

    if res_daily_df is not None and not res_daily_df.empty:
        st.error(f'No data found on {str(input_date)}')
        return

    for tbl in [res_df, res_df_2]:
        if (tbl is not None) and not tbl.empty:
            st.error(msg['error_prob_rows'])
            st.write(tbl)
            error_flag = True

    if not error_flag:
        st.success(msg['success'])


def show_res_daily_view(selected: List[str],
                        result_dict_daily: Dict['str', pd.DataFrame]) -> None:
    """
    Display result for Daily View

    Parameters
    ----------
    selected : List[str]
        Selected tables by the user.
    result_dict_daily : Dict['str', pd.DataFrame]
        Dictionary storing problematic entries for a given date.

    Returns
    -------
    None

    """
    if table_label['bmprices'] in selected:
        show_res_df(table_label['bmprices'],
                    result_dict_daily['bmprices'])
    if table_label['portreturn'] in selected:
        show_res_df(table_label['portreturn'],
                    result_dict_daily['portreturn'])
    if table_label['portholding'] in selected:
        show_res_df(table_label['portholding'],
                    result_dict_daily['portholding_is_daily'],
                    result_dict_daily['portholding'])
    if table_label['bmc_monthly'] in selected:
        show_res_df(table_label['bmc_monthly'],
                    result_dict_daily['bmc_monthly_is_monthly'],
                    result_dict_daily['bmc_monthly'])
    if table_label['univsnapshot'] in selected:
        show_res_df(table_label['univsnapshot'],
                    result_dict_daily['univsnapshot_is_monthly'],
                    result_dict_daily['univ_notin_id'],
                    result_dict_daily['univsnapshot'])
    if table_label['div_ltm'] in selected:
        show_res_df(table_label['div_ltm'],
                    result_dict_daily['div_ltm_is_monthly'],
                    result_dict_daily['div_ltm'])


# =============================================================================
# Code
# =============================================================================

# Set page layout
st.set_page_config(layout="wide")
st.title('Data Quality Checker')

# Load data
data = {'portholding': load_data(query_portholding),
        'bmprices': load_data(query_bmprices),
        'portreturn': load_data(query_portreturn),
        'bmc_monthly': load_data(query_bmc_monthly),
        'div_ltm': load_data(query_div_ltm),
        'univsnapshot': load_data(query_univsnapshot),
        'adjpricet': load_data(query_adjpricet),
        'holiday': load_data(query_holiday)}

data['div_ltm'].columns = ['rdate' if x == 'date' else x
                           for x in data['div_ltm'].columns]


# Side bar
st.sidebar.subheader(header['other_view'])
date_view = st.sidebar.checkbox(checkbox_label['date_view'])

st.sidebar.subheader(header['sum_view'])
is_us_holiday = st.sidebar.checkbox(checkbox_label['us_holiday'], value=True)
is_cad_holiday = st.sidebar.checkbox(checkbox_label['cad_holiday'])

# Sum view
st.header(header['sum_view'])

# '%d' here refers to C-style integer format, not date format.
input_year = st.number_input(
    msg['year_selection'], min_value=1990, max_value=datetime.now().year,
    value=datetime.now().year, format='%d')

holiday_date = get_holiday(input_year, data['holiday'],
                           is_us_holiday, is_cad_holiday)

selected = st.multiselect(msg['multiselect_table'],
                          lst_tables, [table_label['portholding']])

show_table_color_ref = st.checkbox(checkbox_label['color_ref'])

if show_table_color_ref:
    st.dataframe(color_df.style.apply(highlight_color, axis=1))

st.info(msg['update_lag'])

result_dict = get_result_tables(selected, input_year, data)
res_table_df = get_result_sum_df(input_year, holiday_date, result_dict)

show_months(input_year, res_table_df, 'January', 'February', 'March')
show_months(input_year, res_table_df, 'April', 'May', 'June')
show_months(input_year, res_table_df, 'July', 'August', 'September')
show_months(input_year, res_table_df, 'October', 'November', 'December')


# Daily view
if date_view:

    st.header('View by Date')
    input_date = st.date_input("Choose a date",
                               min_value=datetime(input_year, 1, 1),
                               max_value=datetime.today())

    result_dict_daily = get_result_daily(input_date, result_dict)
    show_res_daily_view(selected, result_dict_daily)
