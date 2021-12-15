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
import sys, os
from pathlib import Path
from datetime import date
import itertools

sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter

st.set_page_config(layout="wide")
#%%
# =============================================================================
# Functions - Import data
# =============================================================================
# @st.cache(allow_output_mutation=True)
def get_all_data(input_year):
    # Queries
    query = {
        'bmc_monthly' : f"""SELECT bm_id,rdate,fsym_id FROM development.dbo.bmc_monthly
                                WHERE bm_id IN ('sp500', 'sptsx')
                                and year(rdate)={input_year}
                                order by bm_id, rdate""",
        'univsnapshot' : f"""SELECT univ_id,rdate,fsym_id
                                FROM fstest.dbo.univsnapshot
                                WHERE univ_id IN ('CANADA', 'US')
                                and year(rdate)={input_year}
                                order by univ_id,rdate""",
        'portholding' : f"""SELECT pid,rdate,secid as fsym_id
                                FROM development.dbo.portholding
                                WHERE company NOT IN ('Placeholder', 'Cash-INVEST-USD', 'Cash-INVEST-CAD')
                                and pid in ('bg001','bg013','bg014')
                                and year(rdate)={input_year}
                                order by pid, rdate""",
        'div_ltm' : f"""SELECT div.fsym_id, exdate, date as rdate, div_type, div_freq
                            FROM fstest.dbo.bg_div AS div
                            LEFT JOIN fstest.dbo.bg_div_ltm AS ltm
                            ON div.exdate = ltm.date
                            AND div.fsym_id = ltm.fsym_id
                            WHERE  div_type ='regular'
                            AND dummy_payment = 0
                            and year(exdate)={input_year}
                            order by fsym_id, exdate
                        """,
        'portreturn' : f"""SELECT * FROM development.dbo.PortReturn
                            where year(rdate)={input_year}
                            order by pid, rdate""",
        'bmprices' : f"""SELECT bm_id,rdate,price_index,total_index
                         FROM development.dbo.BMPrice
                         where year(rdate)={input_year}
                         order by bm_id,rdate""",
        'adjpricet' : """SELECT [fsym_id] FROM [FSTest].[dbo].[AdjustedPriceTickers]""",
        'bg_div': """SELECT *
                        FROM [FSTest].[dbo].[BG_Div]
                        WHERE fsym_id IN (SELECT DISTINCT fsym_id FROM [FSTest].[dbo].[BG_Div] WHERE div_type='suspension')""",
        'holiday' : f"""SELECT [fref_exchange_code] ,[holiday_date] ,[holiday_name]
                        FROM [FSTest].[ref_v2].[ref_calendar_holidays]
                        WHERE fref_exchange_code IN ('NYS', 'TSE')
                        and year(holiday_date)={input_year}
                        order by fref_exchange_code, holiday_date"""
        }
    data = {}
    loader = DataImporter(verbose=False)
    for tbl_name, query in query.items():
        # print(tbl_name)
        data[tbl_name] = loader.load_data(query)
    return data
# =============================================================================
# Functions - Check Data Quality
# =============================================================================

# @st.cache
def check_existance(df:pd.DataFrame,
                    dates:pd.Series,
                    month_ends:pd.Series,
                    freq:str,
                    group_key:str=None) -> pd.DataFrame:
    """
    Returns business dates not included in input df

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that we need to check.
    dates : pd.Series
        List of businessd days in a year.
    month_ends : pd.Series
        List of month end days.
    freq : str
        Monthly or Daily or None.
    group_key : str, optional
        The identifier column for benchmark ID, portfolio IDs.
        The default is None.

    Returns
    -------
    res : pd.DataFrame
         Includes the business dates on which we don't have data.

    """
    # df = df[df['rdate'].dt.month!=1]
    if freq == 'monthly':
        results = df.groupby(group_key)['rdate'].\
            apply(lambda x: month_ends[~month_ends.isin(x.values)].values).to_frame('rdate')
    else:
        results = df.groupby(group_key)['rdate'].\
            apply(lambda x: dates[~dates.isin(x.values)].values).to_frame('rdate')

    results = results.explode('rdate').reset_index()
    if results['rdate'].isnull().all():
        return pd.DataFrame()
    results = results[results['rdate'].notnull()].\
            assign(count=1).\
            pivot(index='rdate',columns=group_key, values='count')
    return results

def check_fsym_id(df:pd.DataFrame,
                  bdays:pd.Series,
                  month_ends:pd.Series,
                  freq:str,
                  group_key:str=None) -> pd.DataFrame:
    """
    Returns business dates in which there are empty fsym_ids

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that we need to check.
    bdays : pd.Series
        List of businessd days in a year.
    month_ends : pd.Series
        List of month end days.
    freq : str
        Monthly or Daily or None.
    group_key : str, optional
        The identifier column for benchmark ID, portfolio IDs.
        The default is None.

    Returns
    -------
    results : pd.DataFrame
         Includes the business dates on which we don't have data.

    """
    # df.at[1,'fsym_id'] = np.nan
    df = df[df['fsym_id'].isnull()]

    if df.shape[0] == 0:
        return pd.DataFrame()

    results = df.groupby(group_key)['rdate'].\
        apply(lambda x: bdays[bdays.isin(x.values)].values).to_frame('rdate')
    results = results.explode('rdate').reset_index().assign(count=1).\
        pivot(index='rdate',columns=group_key, values='count')
    return results

# def check_div_screen(check_point:datetime.date,
#                       calendar:pd.DataFrame)->pd.DataFrame:
#     """
#     Check if the dividend screen dataset is updated every week.

#     Parameters
#     ----------
#     check_point : datetime.date

#     calendar : pd.DataFrame
#         Yearly Calendar.

#     Returns
#     -------
#     pd.DataFrame
#         DESCRIPTION.

#     """
#     directory = Path(r'C:\Users\Chang.Liu\Documents\dev\data_update_checker\output')
#     tmp = {}
#     for fname in os.listdir(directory):
#         if 'prq' in fname or 'parquet' in fname:
#             tmp[(directory/Path(fname)).stem] = \
#                 datetime.fromtimestamp(os.stat(directory/Path(fname)).st_mtime).date()
#     res_div_screen = pd.DataFrame(tmp,index=['mdate']).T
#     #dividend screen is update on every sunday.
#     last_sunday = calendar.loc[(calendar['weekday']==6) & (calendar['date']<check_point),'date'].max()
#     return res_div_screen[res_div_screen['mdate']<last_sunday]

def get_result_tables(today:str,
                      input_year: str,
                      data: Dict[str, pd.DataFrame],
                      calendar:pd.DataFrame,
                      table_label:dict) -> Dict[str, pd.DataFrame]:
    """
    Find errors of every table based on the input year and input table


    Parameters
    ----------
    today : datetime.datetime
        Today's date.
    selected : List[str]
        Tables selected in the Streamlit multiselect widget.
    input_year : str
        Year we need to check.
    data : Dict[str, pd.DataFrame]
        Data loaded from DB.
    calendar: pd.DataFrame
        The yearly calendar including week number, holiday and business day.
    Returns
    -------
    result_dict : Dict[str, pd.DataFrame]
        Error entries in data after performing the checks.

    """
    check_point = today + pd.offsets.Day(-1)
    bdays = calendar.loc[(calendar['bday']) & (calendar['date']<check_point), 'date']
    month_ends = calendar.loc[calendar['date']==(calendar['date'] + pd.offsets.MonthEnd(0)),'date']
    month_ends = month_ends[month_ends < check_point]

    group_key_mapping={'portholding':'pid', 'portreturn':'pid',
                       'bmprices':'bm_id', 'bmc_monthly':'bm_id',
                       'univsnapshot':'univ_id'}
    results = {}
    # check missing data on a specific day
    for tbl in list(table_label.keys())[:-4]:
        results[tbl] = {'no_data':check_existance(data[tbl],
                                                  bdays, month_ends,
                                                  table_label[tbl]['frequency'],
                                                  group_key=group_key_mapping[tbl])}
    # check if fsym_id is missing
    for tbl in ['portholding', 'bmc_monthly']:
        results[tbl]['no_fsym_id'] = check_fsym_id(data[tbl],
                                                   bdays,month_ends,
                                                   table_label[tbl]['frequency'],
                                                   group_key=group_key_mapping[tbl])

    # Filters dataframe for rows not also in Adj Price table
    results['univsnapshot']['no_adjprice'] = \
        pd.Series(list(set(data['univsnapshot']['fsym_id']).\
                       difference(set(data['adjpricet']['fsym_id'].values))), dtype='float64')
    # check if the universe changes dramatically from the previous month
    counts = data['univsnapshot'].groupby('univ_id')['rdate'].value_counts().sort_index()
    counts = counts.to_frame('count').groupby('univ_id')['count'].diff() / counts
    results['univsnapshot']['universe_size'] = counts[counts>0.01]

    # check if the datasets for dividend screen is updated on a weekly basis.
    # results['div_screen'] = {'no_data':check_div_screen(check_point, calendar)}
    
    df_div = data['bg_div']
    grouped = df_div.groupby('fsym_id')
    df_div = df_div[grouped.cumcount(ascending=False) > 0]
    st.write(df_div)
    idx_div_ini = df_div[df_div['div_type']=='suspension'].index+1
    st.write(idx_div_ini)
    div_ini_true = df_div[df_div.index.isin(idx_div_ini)]
    st.write(div_ini_true)
    num_bad = 11
    idx_div_ini_true = div_ini_true[~div_ini_true['div_initiation']].index
    num_bad = idx_div_ini_true.size
    res = df_div[df_div.index.isin(idx_div_ini_true) | df_div.index.isin(idx_div_ini_true-1)]
    results['bg_div'] = {'no_div_init_flag': res.rename(columns= {'exdate': 'rdate'})}
    st.write(results['bg_div']['no_div_init_flag'])
    return results

def summary_stats(results:Dict)->Dict:
    """
    Generate an overview of the count of quality issues for each table.

    Parameters
    ----------
    results : Dict
        DESCRIPTION.

    Returns
    -------
    summary : Dict
        DESCRIPTION.

    """
    summary = {}
    for tbl in results.keys():
        no_fsym_id = results[tbl].get('no_fsym_id',pd.DataFrame()).shape[0]
        no_data = results[tbl].get('no_data',pd.DataFrame()).shape[0]
        summary[tbl] = {'Total Number of Quality Issues':no_fsym_id+no_data,
                        'Missing Data':no_data,
                        'Missing Fsym_id':no_fsym_id
                        }
    tbl = 'univsnapshot'
    summary[tbl]['Total Number of Quality Issues'] += \
        results[tbl]['no_adjprice'].shape[0] + results[tbl]['universe_size'].shape[0]
    summary[tbl]['Missing Adjusted Price Ticker'] = results[tbl]['no_adjprice'].shape[0]
    summary[tbl]['Dramatic Universe Size Change'] = results[tbl]['universe_size'].shape[0]
    
    tbl = 'bg_div'
    summary[tbl]['Total Number of Quality Issues'] += \
        results[tbl]['no_div_init_flag'].shape[0] // 2
    summary[tbl]['Missing Dividend Initiation Flag'] = \
        results[tbl]['no_div_init_flag'].shape[0] // 2 
    return summary


# @st.cache
def get_holiday(input_year: int,
                holiday: pd.Series,
                is_us_holiday: bool,
                is_cad_holiday: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """
        Return holidays in the given year based on which holiday calendar is selected
        US holidays are included by default.

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
    if not is_cad_holiday:
        holiday = holiday[holiday['fref_exchange_code'] != 'TSE']
    return holiday['holiday_date'].drop_duplicates()

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
def year_cal(input_year: int,
             holidays:pd.Series) -> pd.DataFrame:
    """
    Return a dataframe of yearly calendar.

    Parameters
    ----------
    input_year : int
        The year that the user selected to view.
    holidays : pd.DataFrame
        Holiday based on the countries selected.

    Returns
    -------
    None.
    """
    dates = pd.date_range(start=datetime(input_year, 1, 1),
                          end=datetime(input_year, 12, 31))
    df = pd.DataFrame({'month': pd.DatetimeIndex(dates).month.astype(int),
                       'weekday': pd.DatetimeIndex(dates).weekday,
                       'day': pd.DatetimeIndex(dates).day.astype(int),
                       'date': dates})

    df['week'] = df['date'].apply(week_of_month)
    df = df.merge(holidays.to_frame('date').assign(holiday_flag=1),
                  on='date',
                  how='left')
    df['bday'] = np.where((df['weekday'].isin([5,6])|df['holiday_flag']==1), False, True)

    return df

def show_months(year_calendar: pd.DataFrame,
                result : Dict,
                freq:str) -> None:
    """
    Show monthly calendars for three months in a row

    Parameters
    ----------
    year_calendar : str
        The year that the user selected to view.
    freq : str
        Update frequency for the table.
    months:List[int]

    Returns
    -------
    None

    """
    for months in [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]:
        cols = st.columns(len(months))
        for i in range(len(months)):
            with cols[i]:
                # getting the month name
                st.header(date(2000, months[i], 1).strftime('%B'))
                show_month_df(year_calendar, result, months[i])

def show_month_df(year_calendar : pd.DataFrame,
                  result: pd.DataFrame,
                  month: int) -> None:
    """
    Show a dataframe with bad dates highlighted for the given month

    Parameters
    ----------
    year_calendar : pd.DataFrame
        The year that the user selected to view.
    res_table_df : pd.DataFrame
        Dataframe contains the error code for each table.
    month : str
        The month we want to display.

    Returns
    -------
    None

    """

    df = year_calendar[year_calendar['month'] == month]
    month_calendar = df.pivot(index='week', columns='weekday', values='day')
    dayOfWeek = {0: 'M', 1: 'T', 2: 'W', 3: 'Th', 4: 'F', 5: 'S', 6: 'Su'}
    month_calendar.columns = [dayOfWeek[i] for i in month_calendar.columns]
    month_calendar = month_calendar.fillna('').astype(str).\
        apply(lambda x: x.str.replace('.0','',regex=False))

    today = [datetime.today().day] if datetime.today().month == month else []
    days = df['date'].to_list()
    weekends = df.loc[df['weekday'].isin([5, 6]),'date'].dt.day.to_list()
    quality_issues = list(set(itertools.chain(*[set(days) & set(r.index) for r in result.values()])))
    quality_issues = sorted([t.day for t in quality_issues])
    holidays = df.loc[df['holiday_flag']==1,'date'].dt.day.to_list()
    st.dataframe(month_calendar.style.apply(highlight_day,
                                            args=[today,
                                                  quality_issues,
                                                  holidays,
                                                  weekends],
                                            axis=1))

def highlight_day(days: pd.Series,
                  today: List[int],
                  quality_issues: List[int],
                  holidays: List[int],
                  weekends: List[int],) -> List[str]:
    """
    Helper function to highlight dates if the day is found in any of the following list:
            1. quality issue
            2. holidays
            3. today's date
            4. weekends

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
    res_colors = []
    for day in days:
        if day == '':
            res_colors.append(None)
            continue
        day = int(day)
        style_str = None
        if day in today:
            style_str = f"background-color: {colors['Today Date']}"
        if day in holidays:
            style_str = f"background-color: {colors['Holidays']}"
        if day in weekends:
            style_str = 'color: lightgrey'
        if day in quality_issues:
            style_str = f"background-color: {colors['Quality Issue']}"
        if day not in set(today+weekends+holidays+quality_issues):
            if len(today) > 0:
                if day > today[0]:
                    style_str = 'color: lightgrey'
        res_colors.append(style_str)
    return res_colors

def highlight_color(row: pd.Series) -> List[str]:
    return  [f"background-color: {colors[row.name]}"]

def summartive_stats(summary, tables):
    st.header('Number of Issues')
    num_of_tables = len(summary.keys())

    metrics = st.columns(num_of_tables)
    for i in range(num_of_tables):
        tbl = list(summary.keys())[i]
        metrics[i].metric(tables[tbl]['description'],
                          summary[tbl]['Total Number of Quality Issues'],
                          delta=None,
                          delta_color="normal")
#%%
# =============================================================================
# Variables
# =============================================================================

table_label = {
    'portholding':{'description':'Portfolio Holding','frequency':'daily'},
    'portreturn':{'description':'Portfolio Return','frequency':'daily'},
    'bmprices':{'description':'BM Prices','frequency':'daily'},
    'bmc_monthly':{'description':'BMC Monthly','frequency':'monthly'},
    'univsnapshot':{'description':'Universe Snapshot','frequency':'monthly'},
    'bg_div':{'description':'BG Div'}, #,'frequency':'history'},
    'div_screen':{'description':'Dividend Screen'},#,'frequency':'monthly'},
    'holiday':'Holiday',
    'today':'Today',
}
# Setup
colors = {'Quality Issue' : 'pink',
          'Holidays' : 'yellow',
          "Today Date" : 'orange'}
color_df = pd.DataFrame(colors,index=['Color']).T
#%%
# Set page layout
if __name__ == '__main__':
    st.title('Data Quality Checker')
    is_us_holiday = st.sidebar.checkbox('Show US Holiday', value=True)
    is_cad_holiday = st.sidebar.checkbox('Show Canadian Holiday',value=True)
    with st.sidebar.expander('Show Color Reference'):
        st.table(color_df.style.apply(highlight_color, axis=1))

    input_year = st.number_input('Select a year',
                                 min_value=1990,
                                 max_value=datetime.now().year,
                                 value=datetime.now().year, format='%d')
    input_year = int(input_year)
    data = get_all_data(input_year)
    holidays = get_holiday(input_year,
                               data['holiday'],
                               is_us_holiday,
                               is_cad_holiday)

    today = date.today()
    year_calendar = year_cal(input_year, holidays)
    results = get_result_tables(today, input_year, data, year_calendar,table_label)
    summary = summary_stats(results)
    summartive_stats(summary, table_label)

    with st.expander('View Details'):
        col1, col2 = st.columns(2)
        selected = col1.radio('Select a table to view details.',
                              list(table_label.keys())[:-2],
                              # index=3,
                              format_func=lambda x:table_label[x]['description'])
        
        st.write(results[selected].items())
        if (summary[selected]['Total Number of Quality Issues'] > 0) and\
        (selected != 'bg_div'):
            tmp = pd.concat([v.assign(issue_type=k).set_index('issue_type',append=True)
                                   for k, v in results[selected].items()]).\
                reorder_levels(['issue_type','rdate']).reset_index()
            tmp['rdate'] = tmp['rdate'].dt.strftime('%Y-%m-%d')
            col2.table(tmp)
#.groupby('fsym_id').count() .groupby('fsym_id')['fsym_id', 'exdate'].count()
        if selected == 'bg_div':
            df = results[selected]['no_div_init_flag']
            st.write(df)
        else: 
            show_months(year_calendar,
                        results[selected],
                        table_label[selected]['frequency'])
