from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras import optimizers
from keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_rows=15

white_wine = pd.read_csv('./wine_data/winequality-white.csv')
red_wine = pd.read_csv('./wine_data/winequality-red.csv')

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc=0)

def generate_data(df: pd.DataFrame, t_r: float):
    total_data = df.to_numpy()
    normalize(total_data)

    np.random.shuffle(total_data)
    n_train_set = int(np.shape(total_data)[0] * t_r)

    x_train = total_data[:n_train_set, :-1]
    y_train = total_data[:n_train_set, -1:]

    x_test = total_data[n_train_set:, :-1]
    y_test = total_data[n_train_set:, -1:]

    return x_train, y_train, x_test, y_test


def normalize(total_data: np.array) -> None:
    for i in range(np.shape(total_data)[1]-1):
        col_zero_base = total_data[:,i] - total_data[:,i].min()
        total_data[:,i] = ( col_zero_base ) / ( col_zero_base.max() ) * 5

x_train, y_train, x_test, y_test = generate_data(white_wine, 0.7)

class ANN_Regression(models.Model):
    def __init__(self, n_in, n_out):
        hidden = layers.Dense(212)
        hidden2 = layers.Dense(128)
        hidden3 = layers.Dense(64)
        hidden4 = layers.Dense(128)
        hidden5 = layers.Dense(256)
        output = layers.Dense(n_out)
        relu = layers.Activation('relu')
        sigmoid = layers.Activation('sigmoid')
        dropout_4 = layers.Dropout(0.48)
        dropout_2 = layers.Dropout(0.2)

        x = layers.Input(shape=(n_in,))
        h = relu(hidden(x))
        h = dropout_4(h)
        h = relu(hidden2(h))
        h = relu(hidden3(h)) 
        h = dropout_2(h)
        h = relu(hidden4(h))
        h = dropout_4(h)
        h = hidden5(h)
        y = output(h)

        adam = optimizers.Adam(lr=0.003, beta_1=0.8)
        super().__init__(x, y)
        self.compile(loss='logcosh', 
                    optimizer=adam,
                    metrics=['accuracy'])

n_in = 11
n_out = 1
model = ANN_Regression(n_in, n_out)

history = model.fit(x_train, y_train, epochs=1000,
                    batch_size=500, validation_split=0.2,
                    verbose=1)

performance_test = model.evaluate(x_test, y_test, batch_size=500)

print(f'\nTest Loss: {performance_test}')

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()




###########################################################


##########################################################


pass
#각 모델의 성능 향상시킬수 있는 방법 적용

###########################################################

##########################################################


pass
# 화이트 와인과 레드 와인을 하나의 모델만 사용하여 분류

###########################################################