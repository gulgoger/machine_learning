# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:23:48 2023

@author: asus_
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.linear_model import LinearRegression


data = pd.read_csv("linear_regression_dataset.csv",sep=";")

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)


linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_pred =linear_regressor.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred,color="red")
plt.show()



#%%-------------------------------

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv("Salary_Data.csv")
dataset.head()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)


y_pred = regressor.predict(x)
y_pred

y_test
