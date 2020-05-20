import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("linear_regression_dataset.csv",
                 sep=";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel="deneyim"
plt.ylabel="maas"
plt.show()

#import sklearn lib
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

#mavi noktaları oluşturduk
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

#fit ettik
linear_reg.fit(x,y) # artık linear reg modelimiz hazır

#predict etme
y_head = linear_reg.predict(x)

plt.plot(x,y_head,color="red")

# yapılan modeli değerlendirme RSQUARE ile

from sklearn.metrics import r2_score

print("r_score2 : ",r2_score(y,y_head) )

#sonuc : r_score2 :  0.9775283164949902
