import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial_regression.csv", sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

#polynomial
polynomial_regression = PolynomialFeatures(degree = 4)

# x kare feature'yi elde etme
x_polynomial = polynomial_regression.fit_transform(x)

#fit
linear_reg = LinearRegression()
linear_reg.fit(x_polynomial,y)

y_head=linear_reg.predict(x_polynomial)

#plot
plt.plot(x,y_head,color="pink", label="poly")
plt.legend()
plt.show()

























