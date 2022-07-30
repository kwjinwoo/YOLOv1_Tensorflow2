from utils.Dataset import DatasetMaker
import os
import argparse


parser = argparse.ArgumentParser(description='Make TFrecord file')
parser.add_argument('--img_dir', type=str, required=True, help='jpg images directory path')
parser.add_argument('--xml_dir', type=str, required=True, help='xml files directory path')
parser.add_argument('--train_txt_file', type=str, required=True, help='train file name list txt file path')
parser.add_argument('--val_txt_file', type=str, required=True, help='val file name list txt file path')
parser.add_argument('--save_dir', type=str, required=True, help='tfrecord file save path')

args = parser.parse_args()


if __name__ == '__main__':
    img_dir = args.img_dir
    xml_dir = args.xml_dir
    train_txt_file = args.train_txt_file
    val_txt_file = args.val_txt_file
    save_dir = args.save_dir

    train_maker = DatasetMaker(img_dir, xml_dir, train_txt_file)
    val_maker = DatasetMaker(img_dir, xml_dir, val_txt_file)

    # train tfrecord
    save_path = os.path.join(save_dir, 'train_PASCAL2007.tfrecord')
    train_maker.make_tfrecord(save_path)

    # val tfrecord
    save_path = os.path.join(save_dir, 'val_PASCAL2007.tfrecord')
    val_maker.make_tfrecord(save_path)
