#!/usr/bin/env python
# coding: utf-8

# # phase 3

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r"C:\rape\20_Victims_of_rape.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.apply(lambda x: sum(x.isnull()),axis=0) # checking missing values in each column of train dataset


# In[8]:


data['Area_Name'].value_counts()


# In[9]:


data['Year'].value_counts()


# In[10]:


data['Rape_Cases_Reported'].value_counts()


# In[11]:


data['Victims_Above_50_Yrs'].value_counts()


# In[12]:


data['Victims_Between_10-14_Yrs'].value_counts()


# In[13]:


data['Victims_Between_14-18_Yrs'].value_counts()


# In[15]:


data['Victims_Between_18-30_Yrs'].value_counts()


# In[17]:


data['Victims_of_Rape_Total'].value_counts()


# In[18]:


data['Victims_Upto_10_Yrs'].value_counts()


# In[22]:


sns.heatmap(data.isnull())
plt.show()


# In[23]:


data.isna().sum()


# In[24]:


data.columns


# In[25]:


data.info()


# In[26]:


data.dtypes


# In[27]:


data.drop(['Subgroup'],axis=1,inplace=True)
data.head()


# In[28]:


data.columns


# In[29]:


data['Rape_Cases_Reported'].unique()


# In[30]:


sns.histplot(data['Area_Name'])
plt.show()


# In[31]:


sns.histplot(data['Rape_Cases_Reported'])
plt.show()


# In[32]:


sns.histplot(data['Victims_of_Rape_Total'])
plt.show()


# In[38]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in data.drop(['Victims_of_Rape_Total'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col}Victims_Above_50_Yrs vs.Victims_of_Rape_Total')
    sns.scatterplot(y=data[col],x=data['Victims_of_Rape_Total'],hue=data['Victims_of_Rape_Total'])
    plt.show()


# In[39]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(),annot=True)


# In[ ]:




