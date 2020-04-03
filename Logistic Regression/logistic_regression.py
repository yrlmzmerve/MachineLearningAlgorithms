import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kanserin iyi huylu olup olmadığını gösteren data
# diagnosis : m->kötü huylu, b-> iyi huylu
data = pd.read_csv("data.csv") 

print(data.info())

#kullanmayacağımız featurlar var bunları drop yapalım
#axis=1 column drop etmek demek, axis=0 row drop etmektir
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

# m=1 ve b=0 diyeceğiz.
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1) #diagnosis hariç tüm sütunlar

#normalization -> 0'la 1 arasına almak
# bu veriler arasındaki büyük farkların modeli kötü etkilemesini önlemek içindir

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values
# x'tei tüm değerler 0'la 1 arasında olmuş oldu.
# datamızdaki tüm veriler 0 ile 1 arasında olduğundan lojistik regr. için uygun data olmuş oldu.

# Train Test Split 
    
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#columnlarla rowun yerilerini ters çevireceğiz
#rowlar feature oldu
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T



