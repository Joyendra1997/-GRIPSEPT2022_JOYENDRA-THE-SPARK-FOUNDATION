#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the data set
data = pd.read_csv('http://bit.ly/w-data')


# In[3]:




data.head()


# In[4]:




data.info() ##checking for null values


# In[5]:




data.describe()


# In[6]:




data.columns #Visualizing the data


# In[7]:




sns.pairplot(data)


# In[8]:




sns.heatmap(data.corr(), annot=True)


# In[9]:




data.corr()


# In[10]:




x = np.asanyarray(data['Hours'])
y = np.asanyarray(data['Scores'])


# In[11]:




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[12]:


lr = LinearRegression()


# In[13]:




lr.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))


# In[14]:




print('Coefficients: ', lr.coef_)
print("Intercept: ", lr.intercept_)


# In[15]:


#plotting the traing set
data.plot(kind='scatter', x='Hours', y='Scores', figsize=(10,5), color='red')
plt.plot(x_train, lr.coef_[0]*x_train + lr.intercept_, color='blue')
plt.title('Training Set')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# In[16]:




predict=lr.predict(np.array(x_test).reshape(-1,1))
predict


# In[17]:


df = pd.DataFrame(np.c_[x_test,y_test,predict], columns=['Hours', 'Actual Score', 'Predicted Scores'])
df


# In[18]:




#testing with given data

hours= [9.25]
pred=lr.predict([hours])
print("NO. of Hours = {}".format(hours))
print("Predicted scores = {}".format(pred[0]))


# In[ ]:




