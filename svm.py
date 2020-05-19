import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# iyi kötü hutlu kanser hücrelerinden olusan 2 sınıflı dataseti
data = pd.read_csv("data.csv")

# kullanılmayan sütunları çıkarma
data.drop(["id","Unnamed: 32"], axis=1, inplace=True)

# Verileri görselleştirme
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean, color="red", alpha=0.4, label="M" )
plt.scatter(B.radius_mean,B.texture_mean, color="green", alpha=0.4, label="B")
plt.legend()
plt.xlabel="radius_mean"
plt.ylabel="texture_mean"
plt.show()

# kategorik değeri int değere dönüştürme
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]

# x ve y değerlerini ayırma

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%%
# normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#%%  SVM 

from sklearn.svm import SVC
svm = SVC(random_state=1)

svm.fit(x_train, y_train)
#svm modeli oluşmus oldu 

print("accuary of svm alg. {}".format(svm.score(x_test,y_test)))

# accuary of svm alg. 0.9590643274853801









