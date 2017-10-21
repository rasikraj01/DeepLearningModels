#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:45:48 2017

@author: rasikraj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

training_set = pd.read_csv('HINDALCO.NS.csv')

training_set = training_set.iloc[:, 1:2].values

#normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#inputs and outputs
X_train = training_set[:4966]  #t
y_train = training_set[1:4967] #t+1

#reshaphing
X_train = np.reshape(X_train, (4966, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()
#input and LSTM layer
regressor.add(LSTM(units=4 , activation='sigmoid', input_shape=(None , 1)))
#putput layer
regressor.add(Dense(units=1))

#compile th RNN

regressor.compile(optimizer='adam', loss='mean_squared_error')


regressor.fit(X_train, y_train, epochs=200, batch_size=32)

from keras.models import load_model

regressor.save('stock_prediction.h5')

#testing


regressor = load_model('stock_prediction.h5')

testset = pd.read_csv('HINDALCO.NS_test.csv')
y_test = testset.iloc[:,1:2].values
 
inputs = y_test
inputs = sc.fit_transform(inputs)

inputs = np.reshape(inputs, (246, 1, 1 ))

y_pred= regressor.predict(inputs)

y_pred = sc.inverse_transform(y_pred)


#plot
plt.plot(y_test, color='red', label='real')
plt.plot(y_pred, color='blue', label='predicted')
plt.title('stock price prediction')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()