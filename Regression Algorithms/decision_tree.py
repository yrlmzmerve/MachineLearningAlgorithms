import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("decision_tree.csv",sep=";",header=None)


x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

x2 = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x).reshape(-1,1)

plt.scatter(x,y,color="blue")
plt.plot(x2,y_head,color="green")
plt.xlabel="tribun"
plt.ylabel="price"
plt.show()
