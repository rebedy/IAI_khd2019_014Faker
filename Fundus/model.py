from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers import BatchNormalization, ReLU


def cnn_sample(in_shape, num_classes=4):    # Example CNN

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=in_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())

    model.add(Conv2D(filters=32, kernel_size=(5, 5),  padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=5,  padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=6,  padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=5,  padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add((ReLU()))

    model.add(Flatten())
    model.add(Dense(100, kernel_initializer=str('glorot_uniform'), activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model
