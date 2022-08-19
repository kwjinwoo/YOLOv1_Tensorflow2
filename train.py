from utils.Dataset import DatasetLoader
from loss import YOLOLoss
from models import network
from tensorflow import keras
import argparse
import tensorflow_addons as tfa


parser = argparse.ArgumentParser(description='Train detector file')
parser.add_argument('--img_size', default=448, type=int, help="image input size")
parser.add_argument('--dataset_dir', required=True, type=str, help="train tfrecord dir")
parser.add_argument('--s', default=7, type=int, help="output grid num")
parser.add_argument('--num_class', default=20, type=int, help="the number of class")
parser.add_argument('--pretrain', default=None, type=str, help="pretrained model weights path")
parser.add_argument('--num_epoch', default=105, type=int, help='train_epoch')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')

args = parser.parse_args()


if __name__ == '__main__':
    img_size = args.img_size
    s = args.s
    num_class = args.num_class

    pretrain = args.pretrain
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    yolo = network.build_model((img_size, img_size, 3))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=40000,
        decay_rate=0.5,
        staircase=True,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = YOLOLoss.get_yolo_loss(img_size, s)
    yolo.compile(loss=loss, optimizer=optimizer)

    loader = DatasetLoader(args.dataset_dir, img_size, s, num_class)
    train_ds, val_ds = loader.get_dataset(batch_size)

    callbacks_list = [keras.callbacks.ModelCheckpoint(
                          filepath='./ckpt/valid_best_yolo',
                          monitor='val_loss',
                          mode='min',
                          save_weights_only=True,
                          save_best_only=True,
                          verbose=1),
                      keras.callbacks.TerminateOnNaN()]

    hist = yolo.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks_list, verbose=1)
    yolo.save_weights('./ckpt/yolo')
