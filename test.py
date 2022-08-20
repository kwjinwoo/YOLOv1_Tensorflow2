import tensorflow as tf
import argparse
from glob import glob
from tqdm import tqdm
import os
import cv2


parser = argparse.ArgumentParser(description='inference file')
parser.add_argument('--model_asset', required=True, type=str, help="YOLO model path")
parser.add_argument('--images_dir', required=True, type=str, help="images dir")

args = parser.parse_args()

class_dict = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor"
}


def get_sample(path, input_shape):
    target_size = input_shape[0]
    image = tf.io.decode_image(tf.io.read_file(path), channels=3)
    image_shape = image.numpy().shape

    input_image = tf.image.resize(image, input_shape[:2])
    input_image = tf.keras.applications.xception.preprocess_input(input_image)
    input_image = tf.expand_dims(input_image, axis=0)

    h_scale = float(image_shape[0] / target_size)
    w_scale = float(image_shape[1] / target_size)

    return input_image, image.numpy(), h_scale, w_scale


def visualize(boxes, names, img, h_scale, w_scale, save_path):
    for box, name in zip(boxes, names):
        pt1 = int(box[1] * w_scale), int(box[0] * h_scale)
        pt2 = int(box[3] * w_scale), int(box[2] * h_scale)
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
        cv2.putText(img, class_dict[name], pt1, color=(255, 0, 0), fontFace=cv2.FONT_ITALIC, fontScale=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    model_path = args.model_asset
    images_list = glob(os.path.join(args.images_dir, "*"))
    print("total images : ", len(images_list))

    save_dir = "./infer"
    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir)

    yolo = tf.keras.models.load_model(model_path)
    model_input = yolo.input.shape[1:].as_list()

    for image_path in tqdm(images_list):
        filename = os.path.basename(image_path)
        input_image, origin_image, h_scale, w_scale = get_sample(image_path, model_input)
        valid_num, nms_boxes, nms_scores, nms_classes = yolo.predict(input_image)

        save_path = os.path.join(save_dir, filename)
        visualize(nms_boxes[0][:valid_num[0]], nms_classes[0][:valid_num[0]],
                  origin_image, h_scale, w_scale, save_path)



