#!/usr/bin/env python
# coding: utf-8

# # Phase 3

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


data = pd.read_csv(r"C:\distric wise crime\01_District_wise_crimes_committed_IPC_2001_2012.csv")


# In[33]:


data


# In[34]:


data.head()


# In[35]:


data.info()


# In[36]:


data.apply(lambda x: sum(x.isnull()),axis=0) # checking missing values in each column of train dataset


# In[37]:


data['STATE/UT'].value_counts()


# In[38]:


# Splitting traing data
X = data.iloc[:, 1: 12].values
y = data.iloc[:, 12].values
X


# In[39]:


y


# In[40]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
X_train


# In[41]:


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0, 5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])

X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
X_train


# In[42]:


y_train


# In[43]:


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for i in range(0, 5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])
X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)
X_test


# In[44]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[45]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[46]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[47]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[48]:


y_pred


# In[49]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred, y_test))


# In[50]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[51]:


# Measuring Accuracy
from sklearn import metrics
print('The accuracy of KNN is: ', metrics.accuracy_score(y_pred, y_test))


# In[52]:


# Making confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[53]:


sns.heatmap(data.isnull())
plt.show()


# In[54]:


data.isna().sum()


# In[55]:


data.columns


# In[56]:


data.info()


# In[57]:


data.dtypes


# In[58]:


data.drop(['ARSON'],axis=1,inplace=True)
data.head()


# In[59]:


data.columns


# In[61]:


data['KIDNAPPING AND ABDUCTION OF OTHERS'].unique()


# In[62]:


sns.histplot(data['DISTRICT'])
plt.show()


# In[63]:


sns.histplot(data['TOTAL IPC CRIMES'])
plt.show()


# In[64]:


sns.histplot(data['KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS'])
plt.show()


# In[65]:


sns.histplot(data['OTHER RAPE'])
plt.show()


# In[67]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in data.drop(['OTHER RAPE'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col}KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS vs. OTHER RAPE')
    sns.scatterplot(y=data[col],x=data['OTHER RAPE'],hue=data['OTHER RAPE'])
    plt.show()


# In[68]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(),annot=True)


# In[ ]:




