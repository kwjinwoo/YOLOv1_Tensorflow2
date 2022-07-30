import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Reshape, Dropout
from keras.models import Sequential, Model


# for pre training
def build_extractor(input_shape):
    feature_extractor = Sequential([
        Conv2D(filters=64, kernel_size=7, strides=2, padding='same', use_bias=False,
               input_shape=input_shape, activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Conv2D(filters=192, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Conv2D(filters=128, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Conv2D(filters=512, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=512, kernel_size=1, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
    ], name='feature_extractor')
    return feature_extractor


def build_head(num_class):
    head = Sequential([
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=(7 * 7 * num_class)),
        Reshape((7, 7, num_class))
    ], name='head')
    return head


def build_yolo(input_shape, num_class, pretrained=False):
    feature_extractor = build_extractor(input_shape)
    if pretrained:
        feature_extractor.load_weights(pretrained)
    head = build_head(num_class)

    yolo = Sequential([
        feature_extractor,
        head
    ])
    return yolo


if __name__ == "__main__":
    pretrain_model = build_extractor((448, 448, 3))
    yolo_model = build_yolo((448, 448, 3), 20)

    print(pretrain_model.summary())
    print(yolo_model.summary())
