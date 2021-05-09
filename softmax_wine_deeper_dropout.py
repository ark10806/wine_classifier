from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras import optimizers
from keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

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

def generate_data(df: pd.DataFrame, t_r: float) -> np.array:
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

global one_hot_codes
one_hot_codes = []
def make_label():
    global one_hot_codes
    tmp = [0] * 10
    for i in range(10):
        tmp[i] = 1
        one_hot_codes.append(tmp.copy())
        tmp[i] = 0

make_label()

def one_hot_enc(y_label: np.array) -> np.array:
    onehot_y = []
    for i in range(np.shape(y_label)[0]):
        idx = int(y_label[i][0])
        onehot_y.append(one_hot_codes[idx])
    
    onehot_y = np.array(onehot_y)
    return onehot_y

x_train, y_train, x_test, y_test = generate_data(white_wine, 0.8)
y_train = one_hot_enc(y_train)
y_test = one_hot_enc(y_test)

x_train_red, y_train_red, x_test_red, y_test_red = generate_data(red_wine, 0.8)
y_train_red = one_hot_enc(y_train_red)
y_test_red = one_hot_enc(y_test_red)

x_train = np.append(x_train, x_train_red, axis=0)
y_train = np.append(y_train, y_train_red, axis=0)
x_test = np.append(x_test, x_test_red, axis=0)
y_test = np.append(y_test, y_test_red, axis=0)

# print(x_train)
# print(y_test)
# exit(0)

# smote = SMOTE(random_state=0)
# x_train_over,y_train_over = smote.fit_resample(x_train, y_train)

# print('SMOTE applied')
# print(x_train.shape, y_train.shape)
# print(x_train_over.shape, y_train_over.shape)
# exit(0)

class ANN_classification(models.Model):
    def __init__(self, n_in, n_h, n_h2, n_h3, n_h4, n_h5, n_h6, n_h7, n_h8, n_h9, n_out):
        hidden = layers.Dense(n_h)
        hidden2 = layers.Dense(n_h2)
        hidden3 = layers.Dense(n_h3)
        hidden4 = layers.Dense(n_h4)
        hidden5 = layers.Dense(n_h5)
        hidden6 = layers.Dense(n_h6)
        hidden7 = layers.Dense(n_h7)
        hidden8 = layers.Dense(n_h8)
        hidden9 = layers.Dense(n_h9)
        hidden10 = layers.Dense(n_h9)
        sigmoid = layers.Activation('sigmoid')
        output = layers.Dense(n_out)
        relu = layers.Activation('softplus')  
        softmax = layers.Activation('softmax')
        dropout_4 = layers.Dropout(0.48)
        dropout_2 = layers.Dropout(0.2)

        x = layers.Input(shape=(n_in,))
        h = relu(hidden(x))
        h = relu(hidden2(h))
        h = dropout_4(h)
        h = relu(hidden3(h))
        h = dropout_2(h)
        h = relu(hidden4(h))
        h = relu(hidden5(h))
        h = relu(hidden6(h))
        h = dropout_2(h)
        h = relu(hidden7(h))
        h = dropout_4(h)
        h = relu(hidden8(h))
        h = dropout_4(h)
        h = relu(hidden9(h))
        h = dropout_2(h)
        h = hidden10(h)
        y = softmax(output(h))

        adam_slow = optimizers.Adam(lr=0.0008, beta_1 = 0.80)

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                    # optimizer='adam',
                    optimizer=adam_slow,
                    metrics=['accuracy'])


n_in = 11
n_h = 128
n_h2 = 96
n_h3 = 64
n_h4 = 32
n_h5 = 16
n_h6 = n_h4
n_h7 = n_h3
n_h8 = n_h2
n_h9 = n_h
n_out = 10
BATCH_SIZE = 200

# n_in = 11
# n_h = 72
# n_h2 = 40
# n_h3 = 32
# n_h4 = 16
# n_h5 = 8
# n_h6 = n_h4
# n_h7 = n_h3
# n_h8 = n_h2
# n_h9 = n_h
# n_out = 10
# BATCH_SIZE = 200

model = ANN_classification(n_in, n_h, n_h2, n_h3, n_h4, n_h5, n_h6, n_h7, n_h8, n_h9, n_out)

# model = ANN_classification(n_in, n_h, n_out)

history = model.fit(x_train, y_train, epochs=1500,
                    batch_size=BATCH_SIZE, validation_split=0.2,
                    verbose=1)

performance_test = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

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