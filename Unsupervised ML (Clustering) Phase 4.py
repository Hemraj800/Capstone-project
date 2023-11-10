#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Importing Libraries,
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[30]:


#Dislay max columns and rows,
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)


# In[31]:


#Load dataset,
data = pd.read_csv(r"C:\crime place 1\17_Crime_by_place_of_occurrence_2014.csv")


# In[32]:


data


# In[33]:


data.describe(include= 'all')


# In[34]:


data.columns


# In[35]:


data.shape


# In[36]:


data.isnull().sum()


# In[37]:


data.dtypes


# In[38]:


data['Residence_Dacoity_Cases reported'].unique()


# In[39]:


data['Total_Theft_Cases reported'].unique()


# In[40]:


#Plotting Scatterplot,
sns.scatterplot(data=data, x='Residence_Dacoity_Cases reported',y='Total_Theft_Cases reported' )


# In[41]:


#Plotting Scatterplot,
sns.scatterplot(data=data, x='ATM_Dacoity_Cases reported',y='ATM_Dacoity_Value of property stolen' )


# In[56]:


#Plotting Distplot
columns = ['ATM_Dacoity_Cases reported','ATM_Dacoity_Value of property stolen']
for columns in columns:
    plt.figure()
    sns.distplot(data[columns])
    sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
    plt.savefig('Distplot.png', box_inches= 'tight')


# In[60]:


#log transformation

columns = ['ATM_Dacoity_Cases reported','ATM_Dacoity_Value of property stolen']
for columns in columns:
    data[columns]= np.log(data[columns])
    plt.figure()
    sns.distplot(data[columns]) 
    plt.savefig('LogTransformation.png', bbox_inches= 'tight')


# In[62]:


#log transformation

columns = ['ATM_Dacoity_Cases reported','ATM_Dacoity_Value of property stolen']
for columns in columns:
    data[columns]= np.log(data[columns])
    plt.figure()
    sns.distplot(data[columns]) 
    plt.savefig('LogTransformation.png', bbox_inches= 'tight')


# In[63]:


data['States/UTs'].sort_values(ascending = True)


# In[64]:


data['States/UTs'].max()


# In[71]:


data_copy = data[['States/UTs', 'Total_Robbery_Cases reported', 'Total_Robbery_Value of property stolen']]
data_copy


# In[73]:


clustering = KMeans(n_clusters=5)
clustering.fit(data_copy[['Total_Robbery_Cases reported','Total_Robbery_Value of property stolen']])
data_copy['Robbery and Robbery cluster'] =clustering.labels_
data_copy.head()


# In[74]:


clustering.cluster_centers_


# In[75]:


clustering.inertia_


# In[76]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data_copy[['Total_Robbery_Cases reported','Total_Robbery_Value of property stolen']])
    intertia_scores.append(kmeans.inertia_)
plt.plot(range(1,11),intertia_scores)


# In[77]:


intertia_scores


# In[78]:


data_copy.groupby(['Robbery and Robbery cluster'])['Total_Robbery_Cases reported', 'Total_Robbery_Value of property stolen'].mean()


# In[81]:


#creating a datframe to get the average of count and weight variables with each cluster
df = pd.DataFrame(data_copy.groupby(['Robbery and Robbery cluster'])['Total_Robbery_Cases reported', 'Total_Robbery_Value of property stolen'].mean())
df


# In[82]:


#Centroids of each cluster 
sns.scatterplot(data=df, x='Total_Robbery_Cases reported',y='Total_Robbery_Value of property stolen')


# In[84]:


centroids =pd.DataFrame(clustering.cluster_centers_)
centroids.columns = ['x','y']


# In[85]:


plt.figure(figsize=(10,8))
plt.scatter(x=centroids['x'],y=centroids['y'],s=100,c='black',marker='*')
sns.scatterplot(data=data_copy, x ='Total_Robbery_Cases reported',y='Total_Robbery_Value of property stolen',hue='Robbery and Robbery cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[86]:


Final_data = pd.concat([data_copy,df]) 
Final_data


# In[ ]:




