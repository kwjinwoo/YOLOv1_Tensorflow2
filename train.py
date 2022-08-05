from utils.Dataset import DatasetLoader
from loss import YOLOLoss
from models import network
from tensorflow import keras
import argparse


parser = argparse.ArgumentParser(description='Train detector file')
parser.add_argument('--img_size', default=448, type=int, help="image input size")
parser.add_argument('--dataset_dir', required=True, type=str, help="train tfrecord dir")
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

    yolo = network.build_model((img_size, img_size, 3), decay)

    optimizer = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    loss = YOLOLoss.get_yolo_loss(img_size, s)
    yolo.compile(loss=loss, optimizer=optimizer)

    loader = DatasetLoader(args.dataset_dir, img_size, s, num_class)
    train_ds = loader.get_dataset(batch_size)

    callbacks_list = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                      keras.callbacks.TerminateOnNaN()]

    hist = yolo.fit(train_ds, epochs=num_epochs, callbacks=callbacks_list)
    yolo.save_weights('./ckpt/yolo')
