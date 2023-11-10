#!/usr/bin/env python
# coding: utf-8

# # phase 3

# In[3]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# In[18]:


df_crime = pd.read_csv(r"C:\crime\17_Crime_by_place_of_occurrence_2001_2012.csv")
df_crime.head(10)


# In[6]:


df_crime=df_crime.dropna()


# In[7]:


#finding correlation b/w all the coefficients sing Pearson Correlation
plt.figure(figsize=(12,10))
cor = df_crime.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[15]:


#since total violent crimes is our target varibale
y = df_crime["TOTAL - Robbery"] 
cor_target = abs(cor["TOTAL - Robbery"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.7] #if correlation > 0.7 then hihgly correlated
relevant_features
     


# In[22]:


x = df_crime[['RESIDENTIAL PREMISES - Dacoity','RESIDENTIAL PREMISES - Robbery','RESIDENTIAL PREMISES - Burglary','RESIDENTIAL PREMISES - Theft','HIGHWAYS - Dacoity','HIGHWAYS - Robbery','HIGHWAYS - Burglary','HIGHWAYS - Theft','RIVER and SEA - Dacoity','RIVER and SEA - Robbery','RIVER and SEA - Theft','RAILWAYS - Dacoity','RAILWAYS - Robbery','RAILWAYS - Burglary','RAILWAYS - Theft','BANKS - Dacoity','BANKS - Robbery','BANKS - Burglary','BANKS - Theft','COMMERCIAL ESTABLISHMENTS - Dacoity','COMMERCIAL ESTABLISHMENTS - Robbery','COMMERCIAL ESTABLISHMENTS - Burglary','COMMERCIAL ESTABLISHMENTS - Theft','OTHER PLACES - Dacoity','OTHER PLACES - Robbery','OTHER PLACES - Burglary','OTHER PLACES - Theft','TOTAL - Dacoity','TOTAL - Robbery','TOTAL - Burglary','TOTAL - Theft']]
y = df_crime['TOTAL - Robbery']


# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25) 
     


# In[24]:


from sklearn.linear_model import LinearRegression
#Fitting the Multiple Linear Regression model
mlr = LinearRegression()  
mlr.fit(x_train, y_train)


# n a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[25]:


print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_))


# In[26]:


#Prediction of test set
y_pred_mlr= mlr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))


# In[27]:


#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head(20)


# In[28]:


#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[ ]:




