# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:56:16 2021

@author: Chang.Liu
"""
    # for error_msg, tbl in zip([f'No data found on {str(input_date)}', 
    #                 msg['error_prob_rows'],
    #                 msg['error_prob_rows']],
    #                 [res_daily_df, res_df, res_df_2]):
    #     if (tbl is not None) and not tbl.empty:
    #         st.error(error_msg)
    #         st.write(tbl)
    #         error_flag = True
    
@st.cache
def find_null(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Filters dataframe for rows that contain null values in input column

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of data that we need to check, loaded from DB
    col : str
        Column of the table we need to check.

    Returns
    -------
    pd.DataFrame
        Rows containing null values for the given column.

    """
    return df[df[col].isnull()]

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
    pd.DataFrame
        Dataframe that contains the error code for each table and date in the
        given year.

    """
    sdate = datetime(input_year, 1, 1)
    edate = datetime(input_year, 12, 31)
    dates = pd.date_range(start=sdate, end=edate)
    res_reason_df = pd.DataFrame(
        {'bmprices': dates.isin(result_dict['bmprices']['result']),
        'portreturn': dates.isin(result_dict['portreturn']['result']),
        'bmc_monthly': dates.isin(result_dict['bmc_monthly']['rdate']),
        'bmc_monthly_is_monthly': 
            dates.isin(result_dict['bmc_monthly_is_monthly']['result']),
        'portholding': dates.isin(result_dict['portholding']['rdate']),
        'portholding_is_daily':  
            dates.isin(result_dict['portholding_is_daily']['result']),
        'univsnapshot': dates.isin(result_dict['univsnapshot']['rdate']),
        'univsnapshot_is_monthly': 
            dates.isin(result_dict['univsnapshot_is_monthly']['result']),
        'univ_notin_id': dates.isin(result_dict['univ_notin_id']['rdate']),
        'div_ltm': dates.isin(result_dict['div_ltm']['date']),
        'div_ltm_is_monthly': 
            dates.isin(result_dict['div_ltm_is_monthly']['result']),
        'holiday': dates.isin(holiday_date)},
        index=dates.date)
    
    res_table_df_col = ['bmprices', 'portreturn', 'bmc_monthly',
                        'portholding', 'univsnapshot', 'div_ltm', 'holiday']
    res_table_df = pd.DataFrame([], columns=res_table_df_col, index=dates.date)
    
    for col in res_reason_df.columns:
        if col in ['bmprices', 'portreturn']:
            res_table_df[col][res_reason_df[col]] = 1 
        elif col == 'bmc_monthly_is_monthly':
            res_table_df['bmc_monthly'][res_reason_df[col]] = 2
        elif col == 'univsnapshot_is_monthly':
            res_table_df['univsnapshot'][res_reason_df[col]] = 2
        elif col == 'div_ltm_is_monthly':
            res_table_df['div_ltm'][res_reason_df[col]] = 2
        elif col in ['bmc_monthly', 'portholding', 'univsnapshot', 'div_ltm']:
            res_table_df[col][res_reason_df[col]] = 3
        elif col == 'univ_notin_id':
            res_table_df['univsnapshot'][res_reason_df[col]] = 4
        elif col == 'holiday':
            res_table_df[col][res_reason_df[col]] = 5
    return res_table_df.fillna(0)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

    # input_year = input_date.year
    # holiday_df, holiday_date = get_holiday(input_year, data['holiday'], is_us_holiday, is_cad_holiday)

    # res_portholding = data['portholding'][data['portholding']['rdate'].dt.year == input_year].copy()
    # res_bmprices = data['bmprices'][data['bmprices']['rdate'].dt.year == input_year].copy()
    # res_portreturn = data['portreturn'][data['portreturn']['rdate'].dt.year == input_year].copy()
    # res_bmc_monthly = data['bmc_monthly'][data['bmc_monthly']['rdate'].dt.year == input_year]
    # res_div_ltm = data['div_ltm'][data['div_ltm']['date'].dt.year == input_year]
    # res_univsnapshot = data['univsnapshot'][data['univsnapshot']['rdate'].dt.year == input_year]

    # # Updated daily
    # res_bmprices['rdate'] = res_bmprices['rdate'].dt.date
    # res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
    # res_portholding['rdate'] = res_portholding['rdate'].dt.date

    # res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate'])
    # res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate'])
    # res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate'])
    # res_portholding = find_null(res_portholding, 'secid')


    # # Updated monthly
    # res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
    # res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    # res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
    # adjpricet_fsym_id = data['adjpricet']['fsym_id'].unique()
    # res_univ_notin_id = res_univsnapshot[~res_univsnapshot['fsym_id'].isin(adjpricet_fsym_id)]
    # res_univsnapshot = find_univsnapshot(res_univsnapshot)
    
    # res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
    # res_div_ltm = find_null(res_div_ltm, 'date')

    # # res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn,res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly = find_res_tables(selected, input_year)
    # result_dict = find_res_tables(selected, input_year, data)
    
    # # res_bmprices = res_bmprices.date
    # # res_bmprices = pd.Series(res_bmprices)
    # # res_portreturn = res_portreturn.date
    # # res_portreturn = pd.Series(res_portreturn)
    # # res_portholding_is_daily = res_portholding_is_daily.date
    # # res_portholding_is_daily = pd.Series(res_portholding_is_daily)

    # res_bmprices_daily = res_bmprices[res_bmprices == input_date]
    # res_portreturn_daily = res_portreturn[res_portreturn == input_date]
    # res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]
    # res_portholding_is_daily_daily = res_portholding_is_daily[res_portholding_is_daily == input_date]

    # res_bmc_monthly_daily = res_bmc_monthly[(res_bmc_monthly['rdate'].dt.year == input_date.year) &
    #                                   (res_bmc_monthly['rdate'].dt.month == input_date.month)]
    # res_bmc_monthly_is_monthly_daily = res_bmc_monthly_is_monthly[(res_bmc_monthly_is_monthly.dt.year == input_date.year) &
    #                                   (res_bmc_monthly_is_monthly.dt.month == input_date.month)]
    # res_div_ltm_daily = res_div_ltm[(res_div_ltm['date'].dt.year == input_date.year) &
    #                           (res_div_ltm['date'].dt.month == input_date.month)]
    # res_div_ltm_is_monthly_daily = res_div_ltm_is_monthly[(res_div_ltm_is_monthly.dt.year == input_date.year) &
    #                           (res_div_ltm_is_monthly.dt.month == input_date.month)]
    # res_univsnapshot_daily = res_univsnapshot[(res_univsnapshot['rdate'].dt.year == input_date.year) &
    #                           (res_univsnapshot['rdate'].dt.month == input_date.month)]
    # res_univsnapshot_is_monthly_daily = res_univsnapshot_is_monthly[(res_univsnapshot_is_monthly.dt.year == input_date.year) &
    #                           (res_univsnapshot_is_monthly.dt.month == input_date.month)]
    # res_univ_notin_id_daily = res_univ_notin_id[(res_univ_notin_id['rdate'].dt.year == input_date.year) &
    #                           (res_univ_notin_id['rdate'].dt.month == input_date.month)]
    return business_dates[~business_dates.isin(res_df_date)].to_frame('result')

def merge_df(df1, df2, df3, df4, df5, df6):
    df1.set_index('rdate',inplace=True)
    df2.set_index('rdate',inplace=True)
    df3.set_index('rdate',inplace=True)
    df4.set_index('rdate',inplace=True)
    df5.set_index('date',inplace=True)
    df6.set_index('rdate',inplace=True)

    df = pd.concat([df1,df2,df3, df4, df5, df6],axis=1,sort=False).reset_index()
    df.rename(columns = {'index':'Col1'})
    st.write(df)
    return df

# TODO Pass the dictionary
def find_res_tables(selected, input_year):
    """Find errors of every table based on the input year and input table """
    if portholding_label in selected:
        res_portholding = portholding[portholding['rdate'].dt.year == input_year]
        res_portholding = res_portholding.copy()
        res_portholding['rdate'] = res_portholding['rdate'].dt.date
        res_portholding_is_daily = check_daily(input_year, holiday_date, res_portholding['rdate']).to_series().dt.date
        res_portholding = find_null(res_portholding, 'secid')
    else:
        res_portholding = pd.DataFrame([], columns = portholding.columns)
        res_portholding_is_daily = pd.DataFrame([], columns = 'result')

    if bmprices_label in selected:
        res_bmprices = bmprices[bmprices['rdate'].dt.year == input_year]
        res_bmprice = res_bmprices.copy()
        res_bmprice['rdate'] = res_bmprices['rdate'].dt.date
        res_bmprices = check_daily(input_year, holiday_date, res_bmprices['rdate']).to_series()
    else:
        res_bmprices = pd.DataFrame([], columns = 'result')

    if portreturn_label in selected:
        res_portreturn = portreturn[portreturn['rdate'].dt.year == input_year].copy()
        res_portreturn['rdate'] = res_portreturn['rdate'].dt.date
        res_portreturn = check_daily(input_year, holiday_date, res_portreturn['rdate']).to_series()
    else:
        res_portreturn = pd.DataFrame([], columns = 'result')
    if bmc_monthly_label in selected:
        res_bmc_monthly = bmc_monthly[bmc_monthly['rdate'].dt.year == input_year]
        res_bmc_monthly_is_monthly = check_monthly(input_year, res_bmc_monthly, 'rdate')
        res_bmc_monthly = find_null(res_bmc_monthly, 'fsym_id')
    else:
        res_bmc_monthly = pd.DataFrame([], columns = bmc_monthly.columns)
        res_bmc_monthly_is_monthly = pd.DataFrame([], columns = 'result')
    if univsnapshot_label in selected:
        res_univsnapshot = univsnapshot[univsnapshot['rdate'].dt.year == input_year]
        res_univsnapshot_is_monthly = check_monthly(input_year, res_univsnapshot, 'rdate')
        res_univsnapshot = find_univsnapshot(res_univsnapshot)
        res_univ_notin_id = not_in_adjpricest(res_univsnapshot)
        res_univsnapshot = res_univsnapshot.merge(res_univ_notin_id, on="rdate", how = 'inner')
    else:
        res_univsnapshot = pd.DataFrame([], columns = univsnapshot.columns)
        res_univsnapshot_is_monthly = pd.DataFrame([], columns = 'result')
        res_univ_notin_id = pd.DataFrame([], columns = univsnapshot.columns)

    if div_ltm_label in selected:
        res_div_ltm = div_ltm[div_ltm['date'].dt.year == input_year]
        res_div_ltm_is_monthly = check_monthly(input_year, res_div_ltm, 'date')
        res_div_ltm = find_null(res_div_ltm, 'date')
    else:
        res_div_ltm = pd.DataFrame([], columns = div_ltm.columns)
        res_div_ltm_is_monthly = pd.DataFrame([], columns = 'result')

    return res_portholding, res_portholding_is_daily, res_bmprices, res_portreturn, res_bmc_monthly, res_bmc_monthly_is_monthly, res_univsnapshot, res_univsnapshot_is_monthly,res_univ_notin_id, res_div_ltm, res_div_ltm_is_monthly

    # res_bmprices_daily = res_bmprices[res_bmprices == input_date]    
    # res_portreturn_daily = res_portreturn[res_portreturn == input_date]
    # res_portholding_daily = res_portholding[res_portholding['rdate'] == input_date]
    # res_portholding_is_daily_daily = res_portholding_is_daily[res_portholding_is_daily == input_date]

    # res_bmc_monthly_daily = res_bmc_monthly[(pd.DatetimeIndex(res_bmc_monthly['rdate']).year == input_date.year) &
    #                                   (pd.DatetimeIndex(res_bmc_monthly['rdate']).month == input_date.month)]
    # res_bmc_monthly_is_monthly_daily = res_bmc_monthly_is_monthly[(pd.DatetimeIndex(res_bmc_monthly_is_monthly).year == input_date.year) &
    #                                   (pd.DatetimeIndex(res_bmc_monthly_is_monthly).month == input_date.month)]
    # res_div_ltm_daily = res_div_ltm[(pd.DatetimeIndex(res_div_ltm['date']).year == input_date.year) &
    #                           (pd.DatetimeIndex(res_div_ltm['date']).month == input_date.month)]
    # res_div_ltm_is_monthly_daily = res_div_ltm_is_monthly[(pd.DatetimeIndex(res_div_ltm_is_monthly).year == input_date.year) &
    #                           (pd.DatetimeIndex(res_div_ltm_is_monthly).month == input_date.month)]
    # res_univsnapshot_daily = res_univsnapshot[(pd.DatetimeIndex(res_univsnapshot['rdate']).year == input_date.year) &
    #                           (pd.DatetimeIndex(res_univsnapshot['rdate']).month == input_date.month)]
    # res_univsnapshot_is_monthly_daily = res_univsnapshot_is_monthly[(pd.DatetimeIndex(res_univsnapshot_is_monthly).year == input_date.year) &
    #                           (pd.DatetimeIndex(res_univsnapshot_is_monthly).month == input_date.month)]
    # res_univ_notin_id_daily = res_univ_notin_id[(pd.DatetimeIndex(res_univ_notin_id['rdate']).year == input_date.year) &
    #                           (pd.DatetimeIndex(res_univ_notin_id['rdate']).month == input_date.month)]

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




res_bmprices = res_bmprices.date
res_bmprices = pd.Series(res_bmprices)
res_portreturn = res_portreturn.date
res_portreturn = pd.Series(res_portreturn)

res_bmprices_daily = res_bmprices[res_bmprices == input_date]    
# res_portreturn_daily = res_portreturn[res_portreturn == input_date]
res_portreturn_daily = pd.DataFrame({'rdate': res_portreturn[res_portreturn == input_date]}, 
                                    columns=portreturn.columns)
# monthly_dates = [datetime(input_year, 1, 1), datetime(input_year, 2, 1), datetime(input_year, 3, 1),
#                  datetime(input_year, 4, 1), datetime(input_year, 5, 1), datetime(input_year, 6, 1),
#                  datetime(input_year, 7, 1), datetime(input_year, 8, 1), datetime(input_year, 9, 1),
#                  datetime(input_year, 10, 1), datetime(input_year, 11, 1), datetime(input_year, 12, 1)]
# monthly_dates = pd.Series(monthly_dates)

if option == 'Summary':
    if st.button('bmc_monthly'):
        show_res(res_bmc_monthly)
    if st.button('portholding'):
        res_portholding = find_null(portholding, 'secid')
        show_res(res_portholding)
    if st.button('PortReturn'):
        res_portreturn = find_daily(in_portreturn, 'pid')
        show_res(res_portreturn)
    if st.button('BMPrice'):
        res_bmprices = find_daily(in_bmprices, 'bm_id')
        show_res(res_bmprices)
    if st.button('univsnapshot'):
        res_univsnapshot = find_univsnapshot(univsnapshot)
        show_res(res_univsnapshot)
    if st.button('bg_div_ltm'):
        res_bg_div_ltm = df_not_equal(div, div_ltm)
        show_res(res_bg_div_ltm)
elif option == 'View by Date':
    input_date = st.date_input("Choose a date")
    res_portholding = portholding[portholding['rdate'] == input_date]
    res_bmprices = bmprices[bmprices['date'] == input_date]
    res_portreturn = portreturn[portreturn['rdate'] == input_date]
    res_bmc_monthly = bmc_monthly[bmc_monthly['date'] == input_date]
    res_div_ltm = div_ltm[div_ltm['date'] == input_date]
    res_div = div[div['date'] == input_date]
    res_univsnapshot = univsnapshot[univsnapshot['date'] == input_date]
    
    
    

def show_date_view(table1, df1, table2, df2, table3, df3):
    
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        if st.button(table1):
            show_res(df1)
    with col2:
        if st.button(table2):
            show_res(df2)
    with col3:
        if st.button(table3):
            show_res(df3)

show_date_view('BMC Monthly', res_bmc_monthly, 'Portfolio Holding', res_portholding, 'Portfolio Return', res_portreturn)
show_date_view('BM Price', res_bmprices, 'Universe Snapshot', res_univsnapshot, 'Div LTM', res_div_ltm)



# Return rows where at least 1 cell in the dataframe is not equal
@st.cache
def df_not_equal(df1, df2):
    df2_sub = df2.iloc[:, 0:2]
    return df2[df1.ne(df2_sub).any(axis=1)]

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


show_cal()

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
