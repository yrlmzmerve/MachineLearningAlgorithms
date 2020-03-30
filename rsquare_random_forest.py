import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("random_forest_regression.csv",
                 sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

randomForest_Reg = RandomForestRegressor(n_estimators = 100,
                                         random_state =42)
randomForest_Reg.fit(x,y)

#predict etme
y_head = randomForest_Reg.predict(x)

#r square

from sklearn.metrics import r2_score

print("r_Score :", r2_score(y,y_head))

# output : r_Score : 0.9798724794092587
# 1 e ne kadar yakınsa sonuc o kadar iyidir