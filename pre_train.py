from models import YOLO
import tensorflow as tf
from tensorflow import keras
from utils.ImageNet import ImageNetLoader
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='pre Train yolo file')
parser.add_argument('--img_size', default=224, type=int, help="image input size")
parser.add_argument('--train_path', required=True, type=str, help="train tfrecord path")
parser.add_argument('--val_path', required=True, type=str, help="val tfrecord path")
parser.add_argument('--num_class', default=1000, type=int, help="the number of class")
parser.add_argument('--num_epoch', default=135, type=int, help='train_epoch')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--decay', default=0.0001, type=float, help='weight decay')

args = parser.parse_args()


if __name__ == '__main__':
    img_size = args.img_size
    train_path = glob(args.train_path)
    val_path = glob(args.val_path)

    decay = args.decay
    num_class = args.num_class
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    loader = ImageNetLoader(train_path, val_path, img_size)
    train_ds, val_ds = loader.get_dataset(batch_size)

    starter_learning_rate = 0.1
    end_learning_rate = 0.01
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5,
        cycle=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy')
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(
        filepath='./ckpt/pretrain',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1)]

    pretrain = YOLO.build_pretrain((img_size, img_size, 3), num_class, decay)
    pretrain.compile(loss=loss, optimizer=optimizer, metrics=[top5, top1])

    hist = pretrain.fit(train_ds, validation_data=val_ds, epochs=num_epoch, callbacks=callbacks_list, verbose=2)
