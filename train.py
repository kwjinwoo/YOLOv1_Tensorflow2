from utils import data_utils
from loss import YOLOLoss
from models import YOLO
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser(description='Train detector file')
parser.add_argument('--img_size', default=448, type=int, help="image input size")
parser.add_argument('--s', default=7, type=int, help="output grid num")
parser.add_argument('--num_class', default=20, type=int, help="the number of class")
parser.add_argument('--pretrain', default=None, type=str, help="pretrained model weights path")
parser.add_argument('--decay', default=0.0005, type=float, help="weight decay")
parser.add_argument('--num_epoch', default=135, type=int, help='train_epoch')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')

args = parser.parse_args()


def lr_scheduler(epoch):
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
    img_size = args.img_size
    s = args.s
    num_class = args.num_class

    pretrain = args.pretrain
    decay = args.decay
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    if pretrain:
        yolo = YOLO.build_yolo((img_size, img_size, 3), s, num_class, decay, pretrain)
    else:
        yolo = YOLO.build_yolo((img_size, img_size, 3), s, decay, num_class)

    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    loss = YOLOLoss.get_yolo_loss(img_size, s)
    yolo.compile(loss=loss, optimizer=optimizer)

    train_ds = data_utils.get_train_dataset(batch_size=batch_size)
    val_ds = data_utils.get_val_dataset(batch_size=batch_size)

    callbacks_list = [keras.callbacks.ModelCheckpoint(
        filepath='./ckpt/yolo',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    ),
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

    hist = yolo.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks_list)
