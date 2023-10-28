# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:59:25 2023

@author: asus_
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv",sep=";")


plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


from sklearn.linear_model import LinearRegression


linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

plt.plot(x,y_head,color="red")


from sklearn.metrics import r2_score

print("r_square score:",r2_score(y, y_head))








