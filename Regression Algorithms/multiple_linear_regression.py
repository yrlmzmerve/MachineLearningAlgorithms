import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")

# deneyim ve yas
# iloc : tüm satıları al, 2. yi al
x = df.iloc[:,[0,2]].values

#maas
y = df.maas.values.reshape(-1,1)
 
multiple_linear_regression = LinearRegression()

#fit etme
multiple_linear_regression.fit(x,y)

# b0 , bias
b0 = multiple_linear_regression.intercept_
print("B0: ", b0)

# b1 ve b2
b1_b2 = multiple_linear_regression.coef_
print("B1,b2",b1_b2)

#predict yapma
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))

# output : array([[11046.35815877],[ 3418.85455609]])
# yani : aynı yasta insanlar biri 11bina lırken diğeri 3bin alıyor
# çünkü biri 10 yıl diğeri 5 yıl tecrübeli