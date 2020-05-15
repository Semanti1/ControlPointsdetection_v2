from csv import reader
import numpy as np
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
with open("predicted_frontal.csv", 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    all_predicted = list(csv_reader)
    k=56
    #data_pred=np.array([[[] for i in range(56)] for j in range(100)])
    data_pred=np.array([all_predicted[0:56]])
    for i in range(1,100):
        temp=all_predicted[k:k+56]
        #print(temp)
        data_pred=np.append(data_pred,[temp],axis=0)
        #print(a)
        k=k+56

print(data_pred.shape)
#print(data_pred)

with open("ground_truth_frontal.csv", 'r') as read_objgt:
    # pass the file object to reader() to get the reader object
    csv_readergt = reader(read_objgt)
    # Pass reader object to list() to get a list of lists
    all_gt = list(csv_readergt)
    k=56
    #data_pred=np.array([[[] for i in range(56)] for j in range(100)])
    data_gt=np.array([all_gt[0:56]])
    for i in range(1,100):
        temp=all_gt[k:k+56]
        #print(temp)
        data_gt=np.append(data_gt,[temp],axis=0)
        #print(a)
        k=k+56
print(data_gt.shape)
model = Sequential()
model.add(TimeDistributed(Dense(10), input_shape=(56,2)))
model.add(LSTM(1000, return_sequences=True))
#model.add(LSTM())
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000, return_sequences=True))
model.add(TimeDistributed(Dense(2)))
model.compile(loss='mse', optimizer='adam')

model.fit(data_pred,data_gt, nb_epoch=1000)
print(model.predict(data_pred))