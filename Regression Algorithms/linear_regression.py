# Libs
import pandas as pd
import matplotlib.pyplot as plt

#import data
# sep=";" ile dataları ayırdık
df = pd.read_csv("linear_regression_dataset.csv",
                 sep=";")

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel="deneyim"
plt.ylabel="maas"
plt.show()

#%% sklearn lib (ml alg var)
import numpy as np

#import sklearn lib
from sklearn.linear_model import LinearRegression

#linear reg model
linear_reg = LinearRegression()

#x=df.deneyim
#y=df.maas
# type(x) ve type(y) Series'dir
# Numpy'a çevirmemiz gerekir
# x.shape() denildiğinde (14,) verir ama sklearn boşlıgu anlamaz
# bosluga 1 yazmak ıcın : reshape(-1,1) kullandık

#mavi noktaları oluşturduk
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values

# fit ettik
linear_reg.fit(x, y)

#prediction
# y = b0+b1*x
# x=0 için y=0
# b0 = y eksenini kestiği nokta : intercept noktası
b0 = linear_reg.predict([[0]])
print("b0:", b0) # b0: [1663.89519747]

#b0 için 2.yol
b0 = linear_reg.intercept_
print("b0:", b0) # b0: 1663.895197474103

# b1
b1 = linear_reg.coef_
print("b1:", b1) #b1: [1138.34819698] (eğim)

# maas = 1663 + 1138*deneyim
# 11 yıllık deneyim için maaş hesaplama
maasyeni = 1663 + 1138*11 # 14181

# 2. yöntem ile maas hesaplama
maas11 = linear_reg.predict([[11]])

#fit edilen line'a bakma

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y,color="g")
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")

#







