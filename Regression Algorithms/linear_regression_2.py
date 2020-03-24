import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("polynomial_regression.csv", sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel="Fiyat"
plt.ylabel="Hız"
plt.show()


#linear regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(x,y) # noktalara en uygun line'ı uydurma

#predict
y_head = linear_reg.predict(x)

plt.plot(x,y_head,color="green")
plt.show()

print("10 milyon tllik araba hızı tahmini:"
      ,linear_reg.predict(1000))



























