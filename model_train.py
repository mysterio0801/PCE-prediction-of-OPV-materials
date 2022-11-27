import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def exel2dict(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    for key, value in data.items():
        data[key] = np.array(list(value.values()))
    return data

def exel2numpy(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    sortednames = sorted(data.keys(), key=lambda x: x.lower())

    data_x = []
    data_y = []

    for key in sortednames:
        if key == 'PCE':
            data_y.append(np.array(list(data[key].values())))
        else:
            data_x.append(np.array(list(data[key].values())))

    return np.array(data_x).T, np.array(data_y).T

inputs = keras.Input(shape=(631,), name='img')
x = layers.Dense(32, activation='relu', kernel_initializer='normal')(inputs)
x = layers.Dense(4, activation='relu', kernel_initializer='normal')(x)
outputs = layers.Dense(1, kernel_initializer='normal')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='model')
model.summary()

keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

x_train, y_train = exel2numpy('train.xlsx')
x_test, y_test = exel2numpy('test.xlsx')

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

x_train = x_train.drop([625, 627], axis = 1)
x_test = x_test.drop([625, 627], axis = 1)

x_train = x_train.astype(float)
x_test = x_test.astype(float)

model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(),
                  metrics=[correlation_coefficient])

history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=100,
                        validation_data=(x_test, y_test)
                        )

model.evaluate(x_test, y_test)