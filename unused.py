# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:56:16 2021

@author: Chang.Liu
"""


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
