import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.wrappers import TimeDistributed

X=np.array([[[1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0]],

       [[0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]]])


Y=np.array([[[ 1],
        [ 3],
        [ 5],
        [ 7],
        [ 9]],

       [[ 2],
        [ 4],
        [ 6],
        [ 8],
        [10]]])


model = Sequential()
model.add(TimeDistributed(Dense(10), input_shape=(5, 2)))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')

model.fit(X,Y, nb_epoch=4000)

print(model.predict(X))