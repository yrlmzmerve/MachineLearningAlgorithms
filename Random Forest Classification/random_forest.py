#import libs
import pandas as pd
import numpy as np

#import dataset
data = pd.read_csv("data.csv")

#Gereksiz columnları cıkaralım
data.drop(["id", "Unnamed: 32"],axis=1, inplace=True)

# string classları int cevirme
data.diagnosis= [1 if each=="M" else 0 for each in data.diagnosis]

# label ve featurelarını ayırma
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

#normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)

#%% Desicion Tree
from sklearn.tree import DecisionTreeClassifier
desicionTree = DecisionTreeClassifier()

desicionTree.fit(x_train, y_train)

print("Desicion Tree Score {}".format(desicionTree.score(x_test, y_test)))

#%% Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators = 100, random_state=1)
randomForest.fit(x_train, y_train)
print("Random Forest Tree Score {}".format(randomForest.score(x_test, y_test)))
