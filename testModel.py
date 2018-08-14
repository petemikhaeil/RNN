import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


model = load_model('model.h5')
train = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

real_price = dataset_test.Open.values.reshape(-1, 1).astype('float64')

# Predicted stock Price
dataset_total = pd.concat((train.Open, dataset_test.Open), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
# Just transform to not regenerate the scaler
inputs = scaler.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0:number_of_inputs])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_of_inputs))
predicted_price = regressor.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)



plt.plot(real_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_price, color='blue', label='Predicted Google Stock Price')
plt.title('Predicted vs Actual Google Stock Price')
plt.xlabel('Time')
plt.xlabel('Stock Opening Price')
plt.legend()
plt.show()
