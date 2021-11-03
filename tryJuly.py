# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:45:11 2021

@author: Chang.Liu
"""

import numpy as np
import streamlit as st

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\july-master\src\july')
import july
from july.utils import date_range

dates = date_range("2020-01-01", "2020-12-31")
data = np.random.randint(0, 14, len(dates))

# july.month_plot(dates, data, month=5) # This will plot only May.
july.calendar_plot(dates, data)
st.pyplot(july.calendar_plot(dates, data))
st.write(july.calendar_plot(dates, data))