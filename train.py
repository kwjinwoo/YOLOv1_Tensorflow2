from utils import data_utils
from loss import loss
from models import network
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser(description='Train detector file')
parser.add_argument('--num_epoch', default=135, type=int, help='train_epoch')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=1e-3, type=float, help='batch_size')

args = parser.parse_args()


def lr_scheduler(epoch, lr):
    if 0 <= epoch <= 75:
        lr = 1e-3 + (9e-3 * (float(epoch) / 75.0))
        return lr
    elif 75 < epoch <= 105:
        lr = 1e-3
        return lr
    else:
        lr = 1e-4
        return lr


if __name__ == '__main__':
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    detector = network.build_model()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    detector.compile(loss=loss.yolo_loss, optimizer=optimizer)

    train_ds = data_utils.get_train_dataset(batch_size=batch_size)
    val_ds = data_utils.get_val_dataset(batch_size=batch_size)

    callbacks_list = [keras.callbacks.ModelCheckpoint(
        filepath='./ckpt/detector',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    ),
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

    hist = detector.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks_list)
