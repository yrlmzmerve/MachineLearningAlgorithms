import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("random_forest_regression.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
randomForest_Reg = RandomForestRegressor(n_estimators = 100, random_state =42)
randomForest_Reg.fit(x,y)

temp = randomForest_Reg.predict([[7.8]])
print("7.8 seviyesinde fiyatın ne kadar olduğu:",temp)
#output : 22.7


x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = randomForest_Reg.predict([[x_]])

# görselleştirme

plt.scatter(x,y,color="pink")
plt.plot(x_,y_head,color="blue")
plt.xlabel="tribun"
plt.ylabel="ucret"
plt.show()

#desicion tree regresyondan farkı 1 yerine 100 tane tree kullanılmasıdır
