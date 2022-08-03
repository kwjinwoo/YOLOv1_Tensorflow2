import tensorflow as tf
from tensorflow import keras
from keras.regularizers import L2


# build model
# return detector model
def build_model(input_shape, decay):
    # feature extractor
    mobilenet = keras.applications.VGG19(include_top=False, input_shape=input_shape, weights='imagenet')

    fc = keras.layers.GlobalAveragePooling2D()(mobilenet.output)   # flatten --> gap
    fc = keras.layers.Dense(7 * 7 * (5 * 2 + 20), kernel_regularizer=L2(decay))(fc)    # predict
    out = keras.layers.Reshape((7, 7, 5 * 2 + 20))(fc)   # reshape

    model = keras.models.Model(mobilenet.input, out)
    return model


# convert output to bounding box
# using non maximum suppression(nms)
class OutputDecoder(keras.layers.Layer):
    def __init__(self, max_detection_per_class=100, max_detection=100, iou_threshold=0.4, score_threshold=0.05):
        super(OutputDecoder, self).__init__()
        self.max_detection_per_class = max_detection_per_class
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detection = max_detection
        self.batch_size = None

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super(OutputDecoder, self).build(input_shape)  # Be sure to call this at the end

    # get corner points of box
    def _convert_to_corner(self, points):
        return tf.concat(
            [points[..., :2] - points[..., 2:] / 2.0, points[..., :2] + points[..., 2:] / 2.0],
            axis=-1,
        )

    # (x, y) --> (y, x)
    def _swap_xy(self, boxes):
        return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

    # convert to origin point
    def _convert_to_origin(self, box):
        # cell nums
        cell_num = tf.linspace(0., 6., 7)   # [0., 1., 2., 3., 4., 5. ,6.]
        y_cells = tf.zeros_like(box[..., 0]) + cell_num
        x_cells = tf.transpose(y_cells, perm=[0, 2, 1])

        converted_x = (x_cells * 64.) + (box[:, :, :, 0] * 64.)  # origin center x
        converted_y = (y_cells * 64.) + (box[:, :, :, 1] * 64.)  # origin center y
        converted_w = tf.minimum(box[..., 2] * 448., 448.)  # origin center w
        converted_h = tf.minimum(box[..., 3] * 448., 448.)  # origin center h
        return tf.stack([converted_x, converted_y, converted_w, converted_h], axis=-1)

    # get confidence score
    def _get_confidence(self, conf, class_):
        confidence_score = class_ * conf
        return confidence_score

    def call(self, pred):
        pred = tf.maximum(pred, 0.)
        # get confidence scores
        classes = tf.maximum(pred[..., 10:], 0.)

        cond = tf.equal(classes, tf.reduce_max(classes, axis=[-1])[..., None])
        classes = tf.where(cond, classes, tf.zeros_like(classes))

        score1 = pred[..., 4, None]
        score1 = tf.reshape(self._get_confidence(score1, classes), (-1, 7 * 7, 20))
        score2 = pred[..., 4 + 5, None]
        score2 = tf.reshape(self._get_confidence(score2, classes), (-1, 7 * 7, 20))
        scores = tf.concat([score1, score2], axis=1)

        # get bounding box1, box2
        box1 = pred[..., :4]
        box2 = pred[..., 5:9]

        # convert origin point
        converted_box1 = tf.reshape(self._convert_to_origin(box1), (-1, 7 * 7, 4))
        converted_box2 = tf.reshape(self._convert_to_origin(box2), (-1, 7 * 7, 4))
        converted_boxes = tf.concat([converted_box1, converted_box2], axis=1)

        # get corner point of box
        corners = self._convert_to_corner(converted_boxes)
        swap_corners = self._swap_xy(corners)   # swap ( x, y --> y, x)

        result = tf.image.combined_non_max_suppression(
                                                     boxes=tf.expand_dims(swap_corners, axis=2),
                                                     scores=scores,
                                                     max_output_size_per_class=self.max_detection_per_class,
                                                     max_total_size=self.max_detection,
                                                     iou_threshold=self.iou_threshold,
                                                     score_threshold=self.score_threshold,
                                                     clip_boxes=False)
        num_val = result.valid_detections   # valid box index
        boxes = result.nmsed_boxes          # nms box
        scores = result.nmsed_scores        # nms confidence score
        class_ = result.nmsed_classes       # nms class

        return num_val, boxes, scores, class_


if __name__ == '__main__':
    detector = build_model((448, 448, 3), 0.0005)
    print(detector.summary())
