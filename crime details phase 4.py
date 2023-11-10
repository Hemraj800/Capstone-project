#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[21]:


df = pd.read_csv('crime.csv')


# In[22]:


df


# In[23]:


df.head()


# In[24]:


df.tail()


# In[25]:


df.sample()


# In[26]:


df1 = df.iloc[:200]
df1.head(200)


# Feature Selection for Clustering Algorithms

# In[27]:


features = ['Population', 'Murder']
X = df1[features].values
y = df1['Rape'].astype(float).values


# In[28]:


features


# In[29]:


X


# In[30]:


y


# Plotting the actual data to vizualize it

# In[31]:


plt.scatter(X[:, 0], X[:, 1], s=50);


# Splitting the data

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Using the elbow method to find the optimal number of clusters

# In[34]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans =KMeans(n_clusters =i, init = 'k-means++', max_iter =300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Vizualising the clusters

# In[37]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)


# In[38]:


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# Metrics Calculation

# In[44]:


print("K-Means")
print("Scores")
print("Accuracy -->",kmeans_accuracy)


# In[61]:


Year = df1['Rape'].astype(int)


# In[62]:


Year.replace('?','0', inplace = True)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
Year.hist()
plt.title('Year')
plt.xlabel('Rape')
plt.ylabel('Frequency')
plt.show()


# In[68]:


Population = df1['Population'].astype(int)
get_ipython().run_line_magic('matplotlib', 'inline')
Population.hist()
plt.title('Rape')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.show


# In[ ]:




