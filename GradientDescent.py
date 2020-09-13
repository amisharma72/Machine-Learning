from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("Iris.csv")
X=data.drop('Species', axis=1)

label=LabelEncoder()
y=data['Species']
X=X.drop('Id',axis=1)
y=label.fit_transform(y)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=1)
encoder = OneHotEncoder(sparse=False)
y_t = y_train.reshape(-1, 1) # Convert data to a single column
y_tt=y_test.reshape(-1,1)
y_train_hot = encoder.fit_transform(y_t)
y_test_hot = encoder.fit_transform(y_tt)

#Model
model = Sequential()
model.add(Dense(8,input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

mini_history=model.fit(X_train, y_train_hot, epochs=30)

stat_history=model.fit(X_train,y_train_hot,batch_size=None, epochs=20)

batch_history=model.fit(X_train,y_train_hot,batch_size=1,epochs=20)

plt.plot(mini_history.history['acc'])
plt.plot(stat_history.history['acc'])
plt.plot(batch_history.history['acc'])
plt.title("Model accuracy,1 Hidden Layer")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['mini-batch', 'stochastic','batch'], loc='upper left')
plt.show()
