#!/usr/bin/env python
# coding: utf-8

# Importing Dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestRegressor


# Importing data

# In[2]:


df = pd.read_csv("C:/Users/shrut/OneDrive/Desktop/gld_price_data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# Correlation:
#     1. positive correlation
#     2. negative correlation

# In[8]:


correlation= df.corr()


# In[9]:


#constructing a heatmap 
plt.figure(figsize=(8,8))

sns.heatmap(correlation , cbar=True , annot = True , square = True ,fmt = '.1f' , annot_kws={'size':8} , cmap='Blues' )


# In[10]:


#correlation values of gold
print(correlation['GLD'])


# In[11]:


#checking the distribution of gld price
sns.distplot(df['GLD'] , color='green')


# In[12]:


#mainly the price is range betwwn 120 


# splitting the features and target

# In[13]:


x = df.drop(["Date" , "GLD"] , axis =1)
y = df["GLD"]


# In[14]:


x


# In[15]:


y


# splitting into train and test data

# In[16]:


x_train , x_test , y_train , y_test= train_test_split(x , y , test_size=0.2 , random_state=2)


# In[17]:


x_train.shape , x_test.shape , y_train.shape , y_test.shape


# Model training : Random Forest Regressor

# In[21]:


algo = RandomForestRegressor(n_estimators=100)


# In[22]:


#training the model
algo.fit(x_train , y_train)


# In[24]:


y_test_prediction=algo.predict(x_test)


# In[25]:


y_train_prediction=algo.predict(x_train)


# In[29]:


plt.plot(y_test_prediction , y_test)
plt.title("test prediction")
plt.xlabel("y_test_prediction")
plt.ylabel("y_test")


# In[30]:


plt.plot(y_train_prediction , y_train)
plt.title("train prediction")
plt.xlabel("y_train_prediction")
plt.ylabel("y_train")


# In[35]:


y_test= list(y_test)


# In[36]:


plt.plot(y_test , color='blue' ,label='Actual Value')
plt.plot(y_test_prediction , color='green' , label='Predicted Value')
plt.title('Actual Price Vs Predicted Price')
plt.xlabel("Number of Values")
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[32]:


training_score=algo.score(x_train , y_train)
print("score on training data : " ,training_score )


# In[33]:


testing_score=algo.score(x_test , y_test)
print("score on testing data : " ,testing_score )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




