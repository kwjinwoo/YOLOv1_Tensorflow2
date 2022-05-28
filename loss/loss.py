import tensorflow as tf


# convert to origin box value(about 224x224 size)
# input : box = [batch_size, 7, 7, 4], obj_cell = [batch_size, 7, 7, 1]
# output : converted boxes = [batch_size, 7, 7, 4]
def convert_to_origin(box, obj_cell):
    batch_size = len(box)   # get batch size

    # cell nums
    cell_num = tf.linspace(0., 6., 7)  # [0., 1., 2., 3., 4., 5. ,6.]
    y_cells = tf.zeros(shape=(batch_size, 7, 7)) + cell_num
    x_cells = tf.transpose(y_cells, perm=[0, 2, 1])

    # except none object cell
    x_cells *= tf.squeeze(obj_cell)
    y_cells *= tf.squeeze(obj_cell)

    converted_x = (x_cells * 32.) + (box[:, :, :, 0] * 32.)   # origin center x
    converted_y = (y_cells * 32.) + (box[:, :, :, 1] * 32.)   # origin center y
    converted_w = box[:, :, :, 2] * 224.   # origin center w
    converted_h = box[:, :, :, 3] * 224.   # origin center h
    return tf.stack([converted_x, converted_y, converted_w, converted_h], axis=-1)


# [x, y, w, h] --> [left_top_x, left_top_y, right_down_x, right_down_y]
# input : points = [batch_size, 7, 7, 4]
# output : corners = [batch_size, 7, 7, 4]
def convert_to_corner(points):
    return tf.concat(
        [points[..., :2] - points[..., 2:] / 2.0, points[..., :2] + points[..., 2:] / 2.0],
        axis=-1,
    )


# calculate intersection over union
def calculate_iou(box1, box2):
    box1_corner = tf.maximum(convert_to_corner(box1), 0.)  # get corner point
    box2_corner = tf.maximum(convert_to_corner(box2), 0.)  # get corner point

    lu = tf.maximum(box1_corner[..., :2], box2_corner[..., :2])    # intersection left top
    rd = tf.minimum(box1_corner[..., 2:], box2_corner[...,  2:])    # intersection right down
    intersection = tf.maximum(tf.zeros_like(rd - lu), rd - lu)    # intersection width, height
    intersection_area = intersection[..., 0] * intersection[..., 1]
    box1_area = (box1[..., 2]) * (box1[..., 3])
    box2_area = (box2[..., 2]) * (box2[..., 3])
    union_area = tf.maximum(
        box1_area + box2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


# get responsible_cell(where the cell idx is responsible)
# output : [batch_size, 7, 7, 2]
def get_responsible_cell(y_true, y_pred, obj_cell):
    box1 = y_pred[..., :4]
    box2 = y_pred[..., 5:9]
    # converted box_info
    converted_box1 = convert_to_origin(box1, obj_cell)
    converted_box2 = convert_to_origin(box2, obj_cell)

    gt_box = y_true[:, :, :, :4]   # ground truth box
    converted_gt_box = convert_to_origin(gt_box, obj_cell)   # converted box_info

    # get iou grid
    box1_iou = calculate_iou(converted_box1, converted_gt_box)   # [batch_size, 7, 7]
    box2_iou = calculate_iou(converted_box2, converted_gt_box)   # [batch_size, 7, 7]

    concat_iou = tf.stack([box1_iou, box2_iou], axis=-1)   # iou grid concat

    # if box1 more responsible or same confidence, and iou not 0
    box1_responsible = tf.where(concat_iou[..., 0] >= concat_iou[..., 1], 1., 0.) * obj_cell
    # if box2 more responsible
    box2_responsible = tf.where(concat_iou[..., 0] < concat_iou[..., 1], 1., 0.) * obj_cell
    return tf.stack([box1_responsible, box2_responsible], axis=-1), concat_iou


# yolo loss
# return bach loss
def yolo_loss(y_true, y_pred):
    # loss param setting
    coord = 5.
    noobj = 0.5
    batch_size = y_true.shape[0]

    # pred split
    x_point_pred = tf.stack([y_pred[:, :, :, 0], y_pred[:, :, :, 0 + 5]], axis=-1)
    y_point_pred = tf.stack([y_pred[:, :, :, 1], y_pred[:, :, :, 1 + 5]], axis=-1)
    w_point_pred = tf.stack([y_pred[:, :, :, 2], y_pred[:, :, :, 2 + 5]], axis=-1)
    w_point_pred = tf.maximum(w_point_pred, 1e-8)   # for sqrt
    h_point_pred = tf.stack([y_pred[:, :, :, 3], y_pred[:, :, :, 3 + 5]], axis=-1)
    h_point_pred = tf.maximum(h_point_pred, 1e-8)   # for sqrt
    c_point_pred = tf.stack([y_pred[:, :, :, 4], y_pred[:, :, :, 4 + 5]], axis=-1)
    class_pred = y_pred[:, :, :, 10:]

    # true split
    x_point_true = y_true[..., 0, None]
    y_point_true = y_true[..., 1, None]
    w_point_true = y_true[..., 2, None]
    h_point_true = y_true[..., 3, None]
    c_point_true = y_true[..., 4, None]
    class_true = y_true[..., 5:]

    obj_cell = tf.where(y_true[:, :, :, 4] == 1., 1., 0.)   # where the cell idx is existed object
    responsible_cell, iou_scores = get_responsible_cell(y_true, y_pred, obj_cell)   # where the cell idx is responsible
    non_responsible_cell = tf.where(responsible_cell == 0., 1., 0.)   # where the cell idx is not responsible

    # x,y loss
    x_loss = tf.square(x_point_pred - x_point_true)
    y_loss = tf.square(y_point_pred - y_point_true)
    xy_loss = (x_loss + y_loss) * responsible_cell
    xy_loss = tf.reduce_sum(xy_loss, [1, 2, 3]) * coord

    # w,h loss
    w_loss = tf.square(tf.sqrt(w_point_pred) - tf.sqrt(w_point_true))
    h_loss = tf.square(tf.sqrt(h_point_pred) - tf.sqrt(h_point_true))
    wh_loss = (w_loss + h_loss) * responsible_cell
    wh_loss = tf.reduce_sum(wh_loss, [1, 2, 3]) * coord

    # confidence loss
    c_loss = tf.square(c_point_pred - iou_scores) * responsible_cell
    c_loss = tf.reduce_sum(c_loss, [1, 2, 3])

    # not confidence loss
    non_c_loss = tf.square(c_point_pred - iou_scores) * non_responsible_cell
    non_c_loss = tf.reduce_sum(non_c_loss, [1, 2, 3]) * noobj

    # class loss
    class_loss = tf.square(class_pred - class_true)
    class_loss = tf.reduce_sum(class_loss, axis=-1) * tf.squeeze(obj_cell)
    class_loss = tf.reduce_sum(class_loss, [1, 2])


    # total loss
    total_loss = xy_loss + wh_loss + c_loss + non_c_loss + class_loss
    total_loss = tf.reduce_mean(total_loss)   # batch loss
    return total_loss


if __name__ == '__main__':
    # temp = tf.zeros(shape=[4, 7, 7, 125]).numpy()
    # # temp_pred = tf.zeros(shape=[4, 7, 7, 130]).numpy()
    # temp_pred = tf.random.normal(shape=[4, 7, 7, 130])
    # temp[:, 2, 3, 4] = 1.
    # temp[:, 3, 3, 4] = 1.
    # temp[:, 2, 3, :4] = [float((112 - 2 * 32.0) / 32.0), float((112 - 3 * 32.0) / 32.0), 1., 1.]
    # temp[:, 3, 3, :4] = [float((112 - 2 * 32.0) / 32.0), float((112 - 3 * 32.0) / 32.0), 1., 1.]
    # # temp_pred[:, 2, 3, :4] = [float((112 - 2 * 32.0) / 32.0),  float((112 - 3 * 32.0) / 32.0), 1., 1.]
    # # temp_pred[:, 3, 3, 5:9] = [float((112 - 2 * 32.0) / 32.0), float((112 - 3 * 32.0) / 32.0), 1., 1.]
    temp = tf.zeros(shape=[1, 7, 7, 25]).numpy()
    temp[0, 3, 2, :5] = [0.83378744, 0.6981132, 0.89373296, 0.9292453, 1.]
    # temp[0, 3, 2, 7] = 1.

    temp_pred = tf.zeros(shape=[1, 7, 7, 30]).numpy()
    # temp_pred[0, 3, 2, :5] = [0, 0, 0, 0, 1.]
    temp_pred[0, 3, 2, :5] = [0.6170765625,  0.6170765625, 0.4256133705, -0.225492066, 0.994193]
    temp_pred[0, 3, 2, 5:10] = [0.7870578125, 0.7870578125, 0.9329433, 0.92087924, 0.9920917]

    print(yolo_loss(temp, temp_pred))
