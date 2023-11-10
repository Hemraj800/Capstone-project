#!/usr/bin/env python
# coding: utf-8

# Data walks through the number of crimes committed against women between 2001 and 2012.
# Shape of data = 10677, 11
# Attributes
# state/ut - India has total 28 states and 8 Union Territories
# district - Each state/ut consists on districts
# year - 2001-2012
# rape - number of rape cases
# kidnapping - number of kidnapping cases
# dowry_deaths - number of dowery deaths
# assault - number of assaulting cases
# insult_to_modesty - number of insult to modesty cases
# cruelty_by_husband - number of women torched by husbands
# trafficking - number of girls importation
# Data_2 Source: https://en.wikipedia.org/wiki/2011_Census_of_India
# Details:
# Data provides census details collected on 2011. Census is conducted every 10 years once in Inida. Recent census was done in 2021.
# Shape of data = 36, 15
# Attributes
# Rank - Highest to lowest population index
# state/ut - State names
# Capital - Capital city of each state
# Type - 2 groups 'states' or 'ut'
# Population - total number people per state
# % of totalpopula-tion[40] - percent of people per state
# Males - number of male population
# Females - number of female population
# SexRatio [41] - sex ratio
# Literacyrate (%) - Literacy rate
# Rural[42] Population - number of people in rural area
# Urban[42] Population - number of people in urban area
# Area[43] (km2) - state area in km
# Density (1/km2) - density of population per km
# DecadalGrowth%(2001–11) - growth in population per state compared to previous census
# Data_3 Source: https://map.igismap.com/gis-data/india/
# Details:
# This dataset provides the gis shape of Indian state border.
# 
# Attributes
# geometry - shape details
# st_nm - State names

# In[5]:


# Import libraries
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Null Hypothesis: Literacy Rate doesn’t affect Crime Rate
Read Data
# In[7]:


# Importing dataset

crimes_df = pd.read_csv(r"C:\file01\42_District_wise_crimes_committed_against_women_2001_2012.csv")


# In[8]:


crimes_df


# In[ ]:




