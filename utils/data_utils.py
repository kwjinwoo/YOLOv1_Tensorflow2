import tensorflow as tf
from tensorflow import keras


# tfrecorder reader
@tf.function
def tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                           "label": tf.io.VarLenFeature(dtype=tf.float32)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_image(image_raw, channels=3)
    label = tf.sparse.to_dense(example["label"])
    label = tf.reshape(label, (7, 7, 25))    # 5 + num_class
    return image, label


# get train dataset
def get_train_dataset(train_type='landmark', data_path='./dataset/train_dataset.tfrecord', batch_size=4):
    ds = tf.data.TFRecordDataset(data_path).map(tfrecord_reader)
    ds = ds.map(preprocessing).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


# get val dataset
def get_val_dataset(train_type='landmark', data_path='./dataset/val_dataset.tfrecord', batch_size=4):
    ds = tf.data.TFRecordDataset(data_path).map(tfrecord_reader)
    ds = ds.map(preprocessing).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


# image preprocessing for mobilenet v2
# -1 ~ 1 range pixel value
def preprocessing(x, y):
    x = keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, dtype=tf.float32))
    return x, y
