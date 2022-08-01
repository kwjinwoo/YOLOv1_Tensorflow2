import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Reshape, Dropout, Input, GlobalAvgPool2D
from keras.models import Sequential, Model
from keras.regularizers import L2


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


def build_head(s, num_class, decay):
    if decay:
        head = Sequential([
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Flatten(),
            Dense(units=4096, activation='relu'),
            Dropout(0.5),
            Dense(units=(s * s * (num_class + 10)), kernel_regularizer=L2(decay)),
            Reshape((s, s, num_class + 10))
        ], name='head')
    else:
        head = Sequential([
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Conv2D(filters=1024, kernel_size=3, padding='same', use_bias=False, activation='relu'),
            Flatten(),
            Dense(units=4096, activation='relu'),
            Dropout(0.5),
            Dense(units=(s * s * (num_class + 10))),
            Reshape((s, s, num_class + 10))
        ], name='head')
    return head


def build_yolo(input_shape, s, num_class, decay=None, pretrained=False):
    feature_extractor = build_extractor(input_shape)
    if pretrained:
        feature_extractor.load_weights(pretrained)
    head = build_head(s, num_class, decay)

    yolo = Sequential([
        feature_extractor,
        head
    ])
    return yolo


def build_pretrain(input_shape, num_class, decay):
    feature_extractor = build_extractor(input_shape)

    inputs = Input(input_shape)
    x = feature_extractor(inputs)
    out = GlobalAvgPool2D()(x)
    out = Dropout(0.5)(out)
    out = Dense(num_class, activation='softmax', kernel_regularizer=L2(decay))(out)

    return Model(inputs, out)


if __name__ == "__main__":
    # pretrain_model = build_extractor((448, 448, 3))
    yolo_model = build_yolo((448, 448, 3), 7, 20)
    pretrain_model = build_pretrain((224, 224, 3), 1000, 0.005)
    print(pretrain_model.summary())
    print(yolo_model.summary())
