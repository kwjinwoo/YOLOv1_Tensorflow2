import tensorflow as tf
import os
import random
import xml.etree.ElementTree as elemTree
from tqdm import tqdm


def txt_to_list(path):
    file_namelist = []
    txt_file = open(path, 'r').readlines()
    for line in txt_file:
        filename = line.strip()
        file_namelist.append(filename)
    random.shuffle(file_namelist)
    return file_namelist


def label_parse(label):
    obj_list = label.getroot().findall('./object')
    names = []
    bndboxes = []
    for obj in obj_list:
        names.append(obj.find('./name').text.encode('utf-8'))
        bndboxes.append(box_parser(obj.find('./bndbox')))
    return names, bndboxes


def box_parser(bndbox):
    xmin = int(bndbox.find('./xmin').text)
    ymin = int(bndbox.find('./ymin').text)
    xmax = int(bndbox.find('./xmax').text)
    ymax = int(bndbox.find('./ymax').text)
    return [xmin, ymin, xmax, ymax]


class DatasetMaker:
    def __init__(self, img_dir, xml_dir, txt_path):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.data_list = txt_to_list(txt_path)

    def load_img(self, name):
        name += '.jpg'
        img = tf.io.decode_jpeg(tf.io.read_file(os.path.join(self.img_dir, name)), channels=3)
        return img

    def load_xml(self, name):
        name += '.xml'
        xml_file = elemTree.parse(os.path.join(self.xml_dir, name))
        return xml_file

    def make_tfrecord(self, save_path):
        with tf.io.TFRecordWriter(save_path) as f:
            for name in tqdm(self.data_list, desc="saving at " + save_path):
                xml_file = self.load_xml(name)
                names, bndboxes = label_parse(xml_file)
                bndboxes = tf.io.serialize_tensor(bndboxes).numpy()
                img = self.load_img(name)
                img = tf.io.encode_jpeg(tf.cast(img, dtype=tf.uint8)).numpy()

                record = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                            'names': tf.train.Feature(bytes_list=tf.train.BytesList(value=names)),
                            'bndboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bndboxes])),
                        }
                    )
                )
                f.write(record.SerializeToString())


class DatasetLoader:
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path

    @tf.function
    def tfrecord_reader(self, example):
        feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                               "names": tf.io.VarLenFeature(dtype=tf.string),
                               "bndboxes": tf.io.VarLenFeature(dtype=tf.string)}

        example = tf.io.parse_single_example(example, feature_description)
        image_raw = tf.sparse.to_dense(example["image"])[0]
        image = tf.io.decode_jpeg(image_raw, channels=3)
        names = tf.sparse.to_dense(example["names"])
        bndboxes = tf.sparse.to_dense(example["bndboxes"])[0]
        bndboxes = tf.io.parse_tensor(bndboxes, out_type=tf.int32)
        return image, names, bndboxes

    def get_dataset(self):
        train_ds = tf.data.TFRecordDataset([self.train_path]).map(self.tfrecord_reader)
        val_ds = tf.data.TFRecordDataset([self.val_path]).map(self.tfrecord_reader)

        return train_ds, val_ds


if __name__ == '__main__':
    maker = DatasetMaker('../dataset/VOCdevkit/VOC2007/JPEGImages', '../dataset/VOCdevkit/VOC2007/Annotations',
                         '../dataset/VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt')
    maker.make_tfrecord(None)
