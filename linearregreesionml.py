#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#linear regression practice


# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[110]:


data=pd.read_csv("C:\\Users\\ravit\\Downloads\\linearregression.csv")


# In[111]:


data


# In[112]:


data.shape


# In[113]:


data.plot(kind='scatter',x='STUDENT',y='AGE')
plt.show()


# In[114]:


data.plot(kind='box')


# In[115]:


data.corr()


# In[116]:


#change to dataframe variables
STUDENT1=pd.DataFrame(data['STUDENT'])
AGE1=pd.DataFrame(data['AGE'])


# In[117]:


STUDENT1


# In[125]:


AGE1


# In[126]:


#BUILD LINEAR MODEL
lm=linear_model.LinearRegression()
model=lm.fit(STUDENT1,AGE1)


# In[127]:


model.coef_


# In[128]:


model.intercept_


# In[129]:


model.score(STUDENT1,AGE1)


# In[130]:


#predict new values
STUDENT1_new=5
AGE1_predict=model.predict(STUDENT1_new)
AGE1_predict


# In[124]:


#predict mode values
X=([67,77,66])
X=pd.DataFrame(X)
Y=model.predict(X)
Y=pd.DataFrame(Y)
df=pd.concat([X,Y],axis=1,keys=['student2_new',student_predicted])
df


# In[108]:


#visualise the result
data.plot(kind='scatter',x='STUDENT',y='AGE')
#plotting the regression line
plt.show(STUDENT1,model.predict(STUDENT1),color='red',linewidth=2)
#plotting the predict
plt.scatter('STUDENT1_new',AGE1_predict,color='black')
plt.plot(X,Y,color='blue',linewidth=5)
plt.show()


# In[ ]:





# In[ ]:




