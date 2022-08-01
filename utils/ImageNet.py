import tensorflow as tf


def augmentation(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, max_delta=0.5)
    return x, y


class ImageNetLoader:
    def __init__(self, train_path, val_path, img_size):
        self.train_path = train_path
        self.val_path = val_path
        self.img_size = img_size

    @tf.function
    def tfrecord_reader(self, example):
        feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                               "names": tf.io.VarLenFeature(dtype=tf.int64),
                               }

        example = tf.io.parse_single_example(example, feature_description)
        image_raw = tf.sparse.to_dense(example["image"])[0]
        image = tf.io.decode_jpeg(image_raw, channels=3)

        names = tf.sparse.to_dense(example["names"])
        return image, names

    def resize_and_normalize(self, x, y):
        x = tf.image.resize(x, (self.img_size, self.img_size)) / 255.
        return x, y

    def get_dataset(self, batch_size):
        train_ds = tf.data.TFRecordDataset([self.train_path], num_parallel_reads=len(self.train_path))
        train_ds = train_ds.map(self.tfrecord_reader).shuffle(buffer_size=10000).map(augmentation)
        train_ds = train_ds.map(self.resize_and_normalize).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.TFRecordDataset([self.val_path], num_parallel_reads=len(self.val_path))
        val_ds = val_ds.map(self.tfrecord_reader).map(self.resize_and_normalize).batch(batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds