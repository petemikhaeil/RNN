import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# Data Prep

train = pd.read_csv('Google_Stock_Price_Train.csv')

# train.Close = train.Close.apply(lambda x: x.replace(',',''))
# training_set = train.iloc[:, 1:2].values
training_set = train[['Open']].values.reshape(-1, 1).astype('float64')
# print(training_set)
scaler = MinMaxScaler(feature_range=(0,1), copy=False)
scaled_training_set = scaler.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output this means for predicting each output it looks back 60 timesteps (e.g. 3 months)
attributes = []
realOpenVals = []
number_of_inputs = 1

for i in range(60, 1258):
    attributes.append(scaled_training_set[i-60:i, :])
    realOpenVals.append(scaled_training_set[i, 0])

attributes = np.array(attributes)
train_real_Open = np.array(realOpenVals).reshape(-1, 1)
#Reshape
# So first dimension correlates to number of rows in your ds, second dimension is number of time steps and final dimension is how many inputs you have
attributes = np.reshape(attributes, (attributes.shape[0], attributes.shape[1], number_of_inputs))
# The above data has 4 columns, ~1000 something rows and 60 rows inside those rows

###BUILDING RNN
regressor = Sequential()

# 3 arguments. Number of units -> number of LSTM cells (sorta like number of nuerons), return_sequences = True when midway point, input_shape is obvs
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(attributes.shape[1], number_of_inputs)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# output Layer
regressor.add(Dense(units=1))

# Compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error')
model = regressor.fit(attributes, realOpenVals, epochs=1, batch_size=32)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

testSet = dataset_test[['Open']].values.reshape(-1, 1).astype('float64')


# Predicted stock Price

scaled_test_set = scaler.transform(testSet)
test_inputs = np.vstack((training_set[-60:], testSet))
# Just transform to not regenerate the scaler
test_attributes = []
for i in range(60, 80):
    test_attributes.append(test_inputs[i-60:i, :])
test_attributes = np.array(test_attributes)
test_attributes = np.reshape(test_attributes, (test_attributes.shape[0], test_attributes.shape[1], number_of_inputs))
predicted_price = regressor.predict(test_attributes)
predicted_price = scaler.inverse_transform(predicted_price)
plt.plot(testSet, color='red', label='Real Google Stock Price')
plt.plot(predicted_price, color='blue', label='Predicted Google Stock Price')
plt.title('Predicted vs Actual Google Stock Price')
plt.xlabel('Time')
plt.xlabel('Stock Opening Price')
plt.legend()
plt.show()
