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

# x_train ve y_train datalarını logistig reg modelini oluşturmak için kullanacağız
# x_test ve y_test datalarıyla modelimizi test edeceğiz

#%% 
# Parameter Initialize and Sigmoid Func

# dimension: 4096 tane pikselimiz yani bizim datamızda 30 tane feature'ımız'
# 30 tane feature varsa 30 tane dimension olması lazım yani dimension = 30
def Initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    # bu komut dimension'a birlik matrix oluşturur ve elemanları 0.01den oluşur
    b = 0.0 #♣float olması için
    return w,b

# w,b = Initialize_weights_and_bias(30)


#%% Sigmoid Function
# formül = f(x) = 1 / (1+ e(üzeri)(-x))    

def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head


#%% Implementing Forward and Backward Propagation
    
def forward_backwar_propagation(w,b,x_train,y_train):
    
    # Forward Propagation
    
# matrix çarpımlarında 1. matrixin columnu ile 2. matrixin row sayısı aynı olmalı
# bu yüzden w'nin T'nu aldık
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1] #normalize etmek gibi bir durum
    
    
    # Backward Propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients 
    # gradient : parametrelerin depolandiği dictionary
   
#%% Implementing Update Parameters 
    
# Weight ile Bias'ı update yapacağız.
    
def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%% Prediction
    
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

