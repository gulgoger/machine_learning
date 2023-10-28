# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:34:00 2023

@author: asus_
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
dataset.head()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

y_pred = regressor.predict(x)
y_pred






