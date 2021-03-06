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
        total_data[:,i] = ( col_zero_base ) / ( col_zero_base.max() )

x_train, y_train, x_test, y_test = generate_data(white_wine, 0.7)

class ANN_Regression(models.Model):
    def __init__(self, n_in, n_h, n_h2, n_h3, n_h4, n_h5, n_out):
        hidden = layers.Dense(n_h)
        hidden2 = layers.Dense(n_h2)
        hidden3 = layers.Dense(n_h3)
        hidden4 = layers.Dense(n_h4)
        hidden5 = layers.Dense(n_h5)
        output = layers.Dense(n_out)
        relu = layers.Activation('relu')
        sigmoid = layers.Activation('sigmoid')

        x = layers.Input(shape=(n_in,))
        h = relu(hidden(x))
        h = relu(hidden2(h))
        h = sigmoid(hidden3(h))
        h = sigmoid(hidden4(h))
        h = hidden5(h)
        y = output(h)


        super().__init__(x, y)

        self.compile(loss='logcosh', 
                    optimizer='adam',
                    metrics=['accuracy'])


n_in = 11
n_h = 1024
n_h2 = 2048
n_h3 = 512
n_h4 = 512
n_h5 = 512
n_out = 1

# n_in = 11
# n_h = 212
# n_h2 = 128
# n_h3 = 64
# n_h4 = 128
# n_h5 = 256
# n_out = 1

model = ANN_Regression(n_in, n_h, n_h2, n_h3, n_h4, n_h5, n_out)

# model = ANN_classification(n_in, n_h, n_out)

history = model.fit(x_train, y_train, epochs=500,
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
#??? ????????? ?????? ??????????????? ?????? ?????? ??????

###########################################################

##########################################################


pass
# ????????? ????????? ?????? ????????? ????????? ????????? ???????????? ??????

###########################################################