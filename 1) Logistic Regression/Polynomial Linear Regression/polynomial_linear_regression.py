# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 00:16:28 2023

@author: asus_
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv",sep =";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)


plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

y_head = lr.predict(x)

plt.plot(x,y_head, color="red")

#print("10 milyon tl lik araba hizi tahmini:",lr.predict(10000))

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)


y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()













