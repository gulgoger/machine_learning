# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:43:59 2023

@author: asus_
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision_tree_regression_dataset.csv",sep=";",header= None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict([[5]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

plt.scatter(x, y, color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("trıbun level")
plt.ylabel("level")
plt.show()

# %%
#DECISION TREE REGRESSİON EXAMPLE 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("IceCreamData.csv",sep=",")

x = dataset["Temperature"].values
y = dataset["Revenue"].values

dataset.head(5)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

y_pred = regressor.predict(x_test.reshape(-1,1))
y_pred

df = pd.DataFrame({"Real Values":y_test.reshape(-1),"Predicted Values":y_pred.reshape(-1)})
df

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x_test,y_test, color="red")
plt.scatter(x_test,y_pred, color="green")
plt.title("Decision Tree Regression")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.show()


plt.plot(x_grid,regressor.predict(x_grid),color="black")
plt.title("Decision Tree Regression")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.show()



























