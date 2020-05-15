import keras.layers as L
import keras.models as M

import numpy

# The inputs to the model.
# We will create two data points, just for the example.
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

data_x = numpy.array([
    # Datapoint 1
    [
        # Input features at timestep 1

        [1, 2],
        # Input features at timestep 2
        [4, 17]
        # Input features at timestep 2

    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [4, 15],
        # Features at timestep 2
        [5, 24]

    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
data_y = numpy.array([
    # Datapoint 1
    [

        [1, 1],
        # Target features at timestep 2
        [4, 16]

    ],
    # Datapoint 2
    [
        # Target features at timestep 1
        [4, 16],
        # Target features at timestep 2
        [5, 25]

    ]
])

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
'''model_input = L.Input(shape=(2, 3))

# This RNN will return timesteps with 4 features each.
# Because return_sequences=True, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 4), which matches the shape of each data point in data_y above.
model_output = L.LSTM(4, return_sequences=True)(model_input)

# Create the model.
model = M.Model(input=model_input, output=model_output)

# You need to pick appropriate loss/optimizers for your problem.
# I'm just using these to make the example compile.
model.compile('sgd', 'mean_squared_error')
# model.compile(loss='mse', optimizer='sgd')
# Train
model.fit(x=data_x, y=data_y,epochs=4000)'''
# model.save('s2s.h5')
print(data_x.shape)
model = Sequential()
model.add(TimeDistributed(Dense(10), input_shape=(2,2)))
model.add(LSTM(2, return_sequences=True))
model.add(LSTM(2, return_sequences=True))
model.add(TimeDistributed(Dense(2)))
model.compile(loss='mse', optimizer='adam')

model.fit(data_x,data_y, nb_epoch=10)
print(model.predict(data_x))