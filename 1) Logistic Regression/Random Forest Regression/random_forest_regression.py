import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head, color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()


#%% ----------------------------------------------------------------- 
# random forest regression example 2


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Salary_Data.csv")

print(data)



x = df.iloc[:,:-1]
y = df.iloc[:,-1:]

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x,y)

y_pred = regressor.predict(np.array([[6.5]]).reshape(1,1))

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x, y, color="blue")

plt.plot(x_grid, regressor.predict(x_grid),color="green")

plt.title("Random Forest Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()






