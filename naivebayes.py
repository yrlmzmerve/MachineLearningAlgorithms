# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data = pd.read_csv("data.csv")
# Data : iyi - kötü huylu kanser hücrelerini göstermektedir
# kullanılmayacak dataları çıkarma
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
#%% Data Visualization
M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]

plt.scatter(M.radius_mean, M.texture_mean, color="red", alpha="0.4", label="M")
plt.scatter(B.radius_mean, B.texture_mean, color="green", alpha="0.2", label="B")
plt.legend()
plt.xlabel="radius_mean"
plt.ylabel="texture_mean"
plt.show()
#%% M _ B değerlerini int değere çevirme
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]
#%% Datayı x _ y ayırma
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
#%% Normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% train_test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(x_train, y_train)

print("Naive Bayes score: {}".format(naiveBayes.score(x_test,y_test)))

# Naive Bayes score: 0.935672514619883
