import tensorflow as tf
import os
import random
from glob import glob
import xml.etree.ElementTree as elemTree
from tqdm import tqdm
import albumentations as A
import numpy as np


class_dict = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}


# convert text file to list
def txt_to_list(path):
    file_namelist = []
    txt_file = open(path, 'r').readlines()
    for line in txt_file:
        filename = line.strip()
        file_namelist.append(filename)
    random.shuffle(file_namelist)
    return file_namelist


# convert class name to int
# get bounding box info(xMin, yMin, xMax, yMax)
def label_parse(label):
    obj_list = label.getroot().findall('./object')
    names = []
    bndboxes = []
    for obj in obj_list:
        names.append(class_dict[obj.find('./name').text])
        bndboxes.append(box_parser(obj.find('./bndbox')))
    return names, bndboxes


# get Box Information
def box_parser(bndbox):
    xmin = int(bndbox.find('./xmin').text)
    ymin = int(bndbox.find('./ymin').text)
    xmax = int(bndbox.find('./xmax').text)
    ymax = int(bndbox.find('./ymax').text)
    return [xmin, ymin, xmax, ymax]


# making Dataset class
class DatasetMaker:
    def __init__(self, img_dir, xml_dir, txt_path):
        self.img_dir = img_dir   # images directory
        self.xml_dir = xml_dir   # xmls directory
        self.data_list = txt_to_list(txt_path)   # file list
        random.shuffle(self.data_list)

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
                try:
                    xml_file = self.load_xml(name)
                except FileNotFoundError as e:
                    continue
                names, bndboxes = label_parse(xml_file)
                bndboxes = tf.io.serialize_tensor(bndboxes).numpy()
                img = self.load_img(name)
                img = tf.io.encode_jpeg(tf.cast(img, dtype=tf.uint8)).numpy()

                record = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                            'names': tf.train.Feature(int64_list=tf.train.Int64List(value=names)),
                            'bndboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bndboxes])),
                        }
                    )
                )
                f.write(record.SerializeToString())


# VOC box format to YOLO format
def convert_to_center(points):
    center_points = tf.concat([points[:2] + (points[2:] - points[:2]) / 2.,
                               points[2:] - points[:2]], axis=0)
    return center_points


# point convert for resized img
def point_adjust(points, img_h, img_w, target_h, target_w):
    x = float(target_w / img_w) * points[..., 0]
    y = float(target_h / img_h) * points[..., 1]
    w = float(target_w / img_w) * points[..., 2]
    h = float(target_h / img_h) * points[..., 3]
    return tf.stack([x, y, w, h], axis=-1)


# augmentation
def aug_fn(img, bndboxes, names):
    data = {"image": img,
            "bboxes": bndboxes,
            "class_indices": names}
    img_shape = img.shape
    transform = get_transformer(img_shape[1], img_shape[0])
    transformed_data = transform(**data)
    transformed_image = transformed_data['image']
    transformed_bboxes = np.array(transformed_data['bboxes'], dtype=np.float32)
    transformed_names = np.array(transformed_data['class_indices'], dtype=np.float32)

    transformed_image = tf.cast(transformed_image, dtype=tf.float32)
    transformed_bboxes = tf.cast(transformed_bboxes, dtype=tf.float32)
    transformed_names = tf.cast(transformed_names, dtype=tf.int64)
    return transformed_image, transformed_bboxes, transformed_names


@tf.function
def transform_augmentation(img, names, bndboxes):
    img, bndboxes, names = tf.numpy_function(func=aug_fn, inp=[img, bndboxes, names],
                                      Tout=[tf.float32, tf.float32, tf.int64])
    img.set_shape([None, None, 3])
    bndboxes.set_shape([None, 4])
    names.set_shape([None])
    return img, names, bndboxes


def get_transformer(w, h):
    h_crop_ratio = np.random.uniform(low=0.25, high=0.9)
    w_crop_ratio = np.random.uniform(low=0.25, high=0.9)
    h_crop = int(h * h_crop_ratio)
    w_crop = int(w * w_crop_ratio)
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=w_crop, height=h_crop, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.2,
            label_fields=['class_indices'],
        ),
    )
    return transform


# Tfrecord file Load class
class DatasetLoader:
    def __init__(self, data_dir, img_size, s, num_class):
        self.val_path = glob(os.path.join(data_dir, 'test*'))   # PASCAL VOC 2007 test
        self.train_path = glob(os.path.join(data_dir, 'trainval*'))
        self.img_size = img_size
        self.s = s
        self.num_class = num_class
        self.cell_len = float(img_size // s)   # one cell length

    @tf.function
    def tfrecord_reader(self, example):
        feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                               "names": tf.io.VarLenFeature(dtype=tf.int64),
                               "bndboxes": tf.io.VarLenFeature(dtype=tf.string)}

        example = tf.io.parse_single_example(example, feature_description)
        image_raw = tf.sparse.to_dense(example["image"])[0]
        image = tf.io.decode_jpeg(image_raw, channels=3)

        names = tf.sparse.to_dense(example["names"])

        bndboxes = tf.sparse.to_dense(example["bndboxes"])[0]
        bndboxes = tf.io.parse_tensor(bndboxes, out_type=tf.int32)
        return image, names, bndboxes

    def resize_and_scaling(self, img, name, bndboxes):
        img_shape = tf.shape(img)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.keras.applications.xception.preprocess_input(img)
        if tf.size(bndboxes) == 0:
            adjust_bndboxes = tf.cast(bndboxes, dtype=tf.float32)
        else:
            adjust_bndboxes = point_adjust(tf.cast(bndboxes, dtype=tf.float32), img_shape[0], img_shape[1],
                                           self.img_size, self.img_size)
        return img, name, adjust_bndboxes

    def get_cell_idx(self, points):
        x = tf.cast(points[0] / self.cell_len, dtype=tf.int32)
        y = tf.cast(points[1] / self.cell_len, dtype=tf.int32)
        return x, y

    def make_grid(self, names, bndboxes):
        grid = tf.zeros(shape=(self.s, self.s, self.num_class + 5))

        for i in tf.range(tf.shape(names)[0]):
            bndbox = bndboxes[i]
            name = names[i]

            center_points = convert_to_center(bndbox)
            x_cell, y_cell = self.get_cell_idx(center_points)

            center_in_cell_x = (center_points[0] - tf.cast(x_cell, dtype=tf.float32) * self.cell_len) / self.cell_len
            center_in_cell_y = (center_points[1] - tf.cast(y_cell, dtype=tf.float32) * self.cell_len) / self.cell_len

            w = center_points[2] / self.img_size
            h = center_points[3] / self.img_size

            class_index = 5 + name

            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, 0]], [center_in_cell_x])
            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, 1]], [center_in_cell_y])
            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, 2]], [w])
            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, 3]], [h])
            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, 4]], [1.])
            grid = tf.tensor_scatter_nd_update(grid, [[x_cell, y_cell, tf.cast(class_index, dtype=tf.int32)]], [1.])
        return grid

    def get_output_grid(self, img, names, bndboxes):
        grid = self.make_grid(names, bndboxes)
        return img, grid

    def get_dataset(self, batch_size):
        train_ds = tf.data.TFRecordDataset(self.train_path, num_parallel_reads=len(self.train_path))
        train_ds = train_ds.map(self.tfrecord_reader).shuffle(buffer_size=5000).map(transform_augmentation)
        train_ds = train_ds.map(self.resize_and_scaling).map(self.get_output_grid).batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.TFRecordDataset(self.val_path)
        val_ds = val_ds.map(self.tfrecord_reader).map(self.resize_and_scaling).map(self.get_output_grid)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds


if __name__ == '__main__':
    maker = DatasetMaker('../dataset/VOCdevkit/VOC2007/JPEGImages', '../dataset/VOCdevkit/VOC2007/Annotations',
                         '../dataset/VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt')
    maker.make_tfrecord(None)
