#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_tr=pd.read_csv(r"C:\Users\HP\Downloads\kidnapping\39_Specific_purpose_of_kidnapping_and_abduction.csv")
df_tr


# In[3]:


df_tr.shape


# In[4]:


df_tr.isnull()


# In[5]:


df_tr.isnull().sum()


# In[6]:


sns.heatmap(df_tr.isnull())
plt.show()


# In[7]:


df_tr.isna().sum()


# In[8]:


df_tr.columns


# In[9]:


df_tr.info()


# In[10]:


df_tr.dtypes


# In[11]:


df_tr.drop(['K_A_Male_Total'],axis=1,inplace=True)
df_tr.head()


# In[12]:


df_tr.columns


# In[13]:


df_tr['Area_Name'].unique()


# In[16]:


df_tr['K_A_Female_18_30_Years'].unique()


# In[17]:


sns.histplot(df_tr['K_A_Female_15_18_Years'])
plt.show()


# In[18]:


sns.histplot(df_tr['K_A_Female_18_30_Years'])
plt.show()


# In[19]:


sns.histplot(df_tr['Area_Name'])
plt.show()


# In[20]:


from sklearn.preprocessing import LabelEncoder # import


# In[21]:


le=LabelEncoder()
for i in df_tr.drop(['Area_Name'],axis=1):
    df_tr[i]=le.fit_transform(df_tr[i])
df_tr


# In[22]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in df_tr.drop(['K_A_Female_18_30_Years'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col} vs. K_A_Female_18_30_Years')
    sns.scatterplot(y=df_tr[col],x=df_tr['K_A_Female_18_30_Years'],hue=df_tr['K_A_Female_18_30_Years'])
    plt.show()


# In[23]:


df_tr.corr()


# In[24]:


plt.figure(figsize=(15,8))
sns.heatmap(df_tr.corr(),annot=True)


# In[ ]:




