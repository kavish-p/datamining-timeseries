#!/usr/bin/env python
# coding: utf-8

# In[67]:


# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[115]:


# data imports
path = 'C:/Users/Kavish/Dropbox/_Master of Data Science - UM/WQD7005 - Data Mining/Assignment/Milestone 5/'
df = pd.read_csv(path + 'data_combined_sources.csv')
df.head()

df['Datetime'] = pd.to_datetime(df.Datetime)
# df = df.set_index('Datetime')
# df = df.sort_values(by='Datetime')


# In[116]:


df.info()


# In[206]:


df.head()


# In[117]:


df.isnull().sum()


# In[127]:


def plot_ts(code):
    temp = df[df['CompanySymbol'] == code]
    temp = temp.set_index('Datetime')
    temp = temp.sort_values(by='Datetime')
    plt.plot(temp['LastPrice'])
    
def filter_company(code):
    temp = df[df['CompanySymbol'] == code]
    temp = temp.set_index('Datetime')
    temp = temp.sort_values(by='Datetime')
    return temp
    
plot_ts('MAYBANK')


# In[122]:


# tweet = pd.get_dummies(df['TweetLabel'],drop_first=True)
# news = pd.get_dummies(df['NewsLabel'],drop_first=True)

df['ChangeLabel'] = df['ChangeLabel'].map( {'Unchanged': 0, 'Decrease': 1, 'Increase': 2} ).astype(int)
df['NewsLabel'] = df['NewsLabel'].map( {'Neutral': 0, 'Negative': 1, 'Positive': 2} ).astype(int)
df['TweetLabel'] = df['TweetLabel'].map( {'Neutral': 0, 'Negative': 1, 'Positive': 2} ).astype(int)

new_df = df.drop(['Datetime','CompanySymbol','OpenPrice','HighPrice','LowPrice','LastPrice','Difference'],axis=1)
new_df.head()


# In[123]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_df.drop('ChangeLabel',axis=1),new_df['ChangeLabel'], test_size=0.30,random_state=101)


# In[128]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[129]:


from sklearn import metrics
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))


# In[130]:


plot_ts('MAYBANK')


# In[131]:


from statsmodels.tsa.arima_model import ARIMA
from random import random

data = filter_company('MAYBANK')
data.head()


# In[132]:


maybank_last_price = data.LastPrice.values
maybank_last_price

model = ARIMA(maybank_last_price, order=(1, 1, 1))
model_fit = model.fit(disp=0)
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
model_fit.summary()


# In[133]:


model_fit.plot_predict(dynamic=False)
plt.show()


# In[203]:


from statsmodels.tsa.stattools import acf


new_data = data[['LastPrice']]

# Create Training and Test
train = new_data.values[:85]
test = new_data.values[-8:]


# # Create Training and Test
# train = maybank_last_price[:83]
# test = maybank_last_price[-10:]

model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  

fc, se, conf = fitted.forecast(10, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc)
lower_series = pd.Series(conf[:, 0])
upper_series = pd.Series(conf[:, 1])


# for i in range(85):
#     test = np.insert(test,0,'NaN', axis=0)

# Plot
plt.figure(figsize=(10,4), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper right', fontsize=8)
plt.show()

