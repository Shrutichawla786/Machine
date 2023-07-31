#!/usr/bin/env python
# coding: utf-8

# importing depandencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("C:/Users/shrut/OneDrive/Desktop/winequality-red.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


#checking the missing values
df.isnull().sum()


# Data Analysis and Visulaization

# In[8]:


#statistical measures of the datasets
df.describe()


# In[9]:


#number of values for each quality
sns.catplot(x='quality' , data=df , kind='count')


# In[16]:


#volatile acidity vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality' , y='volatile acidity' , data=df)
plt.show()


# In[18]:


#the ablove graph tells us that volatile acidity is inversly proportional to quality


# In[17]:


#critic acidity vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality' , y='citric acid' , data=df)
plt.show()


# In[19]:


#the above graph tells us that citric acid is directly proportional to quality


# correlation
# 1. positive correlation
# 2. negative correlation

# In[20]:


correlation= df.corr()


# In[25]:


#constructing a heatmap to understand the correlation between difffernt columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation , cbar=True , square = True , fmt ='.1f' ,annot=True , annot_kws={'size' :8}, cmap='Blues')
plt.show()


# Data Preprocessing

# In[28]:


x = df.drop(columns='quality' ,axis=1 )
y = df['quality']


# In[32]:


print(x)


# Label Binarization

# In[36]:


y = df['quality'].apply(lambda y_value:1 if y_value>=7 else 0)


# In[37]:


print(y)


# Train and Test split

# In[38]:


x_train , x_test , y_train , y_test =train_test_split(x , y , test_size=0.2 , random_state=2)


# In[39]:


x_train.shape , x_test.shape , y_train.shape , y_test.shape


# In[42]:


algo = RandomForestClassifier()


# In[43]:


algo.fit(x_train , y_train)


# In[44]:


y_predict= algo.predict(x_test)


# In[45]:


y_predict


# In[47]:


plt.plot(y_test , y_predict)
plt.show()


# In[53]:


train_data=algo.score(x_train , y_train)
print("Train_data_accuracy = " , train_data)


# In[54]:


test_data=algo.score(x_test , y_test)
print("Test_data_accuray = " , test_data)


# Building a predictive sysytem

# In[59]:


input_data=(8.1,0.56,0.28,1.7,0.368,16.0,56.0,0.9968,3.11,1.28,9.3)
#changning the input data into numpy array
input_data_np= np.asarray(input_data)
#reshape the data
input_data_reshape= input_data_np.reshape(1,-1)

prediction = algo.predict(input_data_reshape)

if(prediction[0]==1):
    print("Quality is good")
else:
    print("Quality is bad")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




