import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    dates, data = generate_data()
    # st.write(dates)
    st.write(data)
    # st.write(type(dates))
    # st.write(type(data))

    fig, ax = plt.subplots(figsize=(6, 10))
    calendar_heatmap(ax, dates, data)
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
    ax.figure.colorbar(im)

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

main()
# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 27 13:57:33 2021

# @author: Chang.Liu
# """
# import streamlit as st

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# # Settings
# years = [2018] # [2018, 2019, 2020]
# weeks = [1, 2, 3, 4, 5, 6]
# days = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
# month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
#                'September', 'October', 'November', 'December']

# def generate_data():
#     idx = pd.date_range('2018-01-01', periods=365, freq='D')
#     return pd.Series(range(len(idx)), index=idx)


# def split_months(df, year):
#     """
#     Take a df, slice by year, and produce a list of months,
#     where each month is a 2D array in the shape of the calendar
#     :param df: dataframe or series
#     :return: matrix for daily values and numerals
#     """
#     df = df[df.index.year == year]


#     # Empty matrices
#     a = np.empty((6, 7))
#     a[:] = np.nan

#     day_nums = {m:np.copy(a) for m in range(1,13)}  # matrix for day numbers
#     day_vals = {m:np.copy(a) for m in range(1,13)}  # matrix for day values

#     # Logic to shape datetimes to matrices in calendar layout
#     for d in df.iteritems():  # use iterrows if you have a DataFrame

#         day = d[0].day
#         month = d[0].month
#         col = d[0].dayofweek

#         if d[0].is_month_start:
#             row = 0

#         day_nums[month][row, col] = day  # day number (0-31)
#         day_vals[month][row, col] = d[1] # day value (the heatmap data)

#         if col == 6:
#             row += 1

#     return day_nums, day_vals


# def create_year_calendar(day_nums, day_vals):
#     fig, ax = plt.subplots(3, 4, figsize=(14.85, 10.5))

#     for i, axs in enumerate(ax.flat):

#         axs.imshow(day_vals[i+1], cmap='viridis', vmin=1, vmax=365)  # heatmap
#         axs.set_title(month_names[i])

#         # Labels
#         axs.set_xticks(np.arange(len(days)))
#         axs.set_xticklabels(days, fontsize=10, fontweight='bold', color='#555555')
#         axs.set_yticklabels([])

#         # Tick marks
#         axs.tick_params(axis=u'both', which=u'both', length=0)  # remove tick marks
#         axs.xaxis.tick_top()

#         # Modify tick locations for proper grid placement
#         axs.set_xticks(np.arange(-.5, 6, 1), minor=True)
#         axs.set_yticks(np.arange(-.5, 5, 1), minor=True)
#         axs.grid(which='minor', color='w', linestyle='-', linewidth=2.1)

#         # Despine
#         for edge in ['left', 'right', 'bottom', 'top']:
#             axs.spines[edge].set_color('#FFFFFF')

#         # Annotate
#         for w in range(len(weeks)):
#             for d in range(len(days)):
#                 day_val = day_vals[i+1][w, d]
#                 day_num = day_nums[i+1][w, d]

#                 # Value label
#                 axs.text(d, w+0.3, f"{day_val:0.0f}",
#                          ha="center", va="center",
#                          fontsize=7, color="w", alpha=0.8)

#                 # If value is 0, draw a grey patch
#                 if day_val == 0:
#                     patch_coords = ((d - 0.5, w - 0.5),
#                                     (d - 0.5, w + 0.5),
#                                     (d + 0.5, w + 0.5),
#                                     (d + 0.5, w - 0.5))

#                     square = Polygon(patch_coords, fc='#DDDDDD')
#                     axs.add_artist(square)

#                 # If day number is a valid calendar day, add an annotation
#                 if not np.isnan(day_num):
#                     axs.text(d+0.45, w-0.31, f"{day_num:0.0f}",
#                              ha="right", va="center",
#                              fontsize=6, color="#003333", alpha=0.8)  # day

#                 # Aesthetic background for calendar day number
#                 patch_coords = ((d-0.1, w-0.5),
#                                 (d+0.5, w-0.5),
#                                 (d+0.5, w+0.1))

#                 triangle = Polygon(patch_coords, fc='w', alpha=0.7)
#                 axs.add_artist(triangle)

#     # Final adjustments
#     fig.suptitle('Calendar', fontsize=16)
#     plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.04)

#     # Save to file
#     plt.savefig('calendar_example.pdf')
#     return fig


# for year in years:
#     df = generate_data()
#     day_nums, day_vals = split_months(df, year)
#     st.pyplot(create_year_calendar(day_nums, day_vals))