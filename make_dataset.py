from utils.Dataset import DatasetMaker
import os
import argparse


parser = argparse.ArgumentParser(description='Make TFRecord file')
parser.add_argument('--dataset_root', type=str, required=True, help='dataset root dir path')
parser.add_argument('--save_dir', type=str, required=True, help='tfrecord file save path')

args = parser.parse_args()


if __name__ == '__main__':
    dataset_root = args.dataset_root
    save_dir = args.save_dir

    # VOC 2007 train val
    img_dir = os.path.join(dataset_root, 'VOC2007/JPEGImages')
    xml_dir = os.path.join(dataset_root, 'VOC2007/Annotations')
    txt_path = os.path.join(dataset_root, 'VOC2007/ImageSets/Main/trainval.txt')
    trainval2007_maker = DatasetMaker(img_dir, xml_dir, txt_path)
    save_path = os.path.join(save_dir, 'trainval_2007.tfrecord')
    trainval2007_maker.make_tfrecord(save_path)

    # VOC 2007 test
    img_dir = os.path.join(dataset_root, 'VOC2007_test/JPEGImages')
    xml_dir = os.path.join(dataset_root, 'VOC2007_test/Annotations')
    txt_path = os.path.join(dataset_root, 'VOC2007_test/ImageSets/Main/test.txt')
    test2007_maker = DatasetMaker(img_dir, xml_dir, txt_path)
    save_path = os.path.join(save_dir, 'test_2007.tfrecord')
    test2007_maker.make_tfrecord(save_path)

    # VOC 2012 train val
    img_dir = os.path.join(dataset_root, 'VOC2012/JPEGImages')
    xml_dir = os.path.join(dataset_root, 'VOC2012/Annotations')
    txt_path = os.path.join(dataset_root, 'VOC2012/ImageSets/Main/trainval.txt')
    trainval2012_maker = DatasetMaker(img_dir, xml_dir, txt_path)
    save_path = os.path.join(save_dir, 'trainval_2012.tfrecord')
    trainval2012_maker.make_tfrecord(save_path)
