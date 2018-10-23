"""OLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

BIN = 2
DIMSCALE = 50.0
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
#    x = DarknetConv2D_BN_Leaky(24, (3,3))(x)
#    x = resblock_body(x, 48, 1)
#    x = resblock_body(x, 96, 2)
#    #x = resblock_body(x, 192, 3)
#    x = resblock_body(x, 192, 4)
#    x = resblock_body(x, 384, 2)
#    x = resblock_body(x, 768, 2)
#    

    return x

def make_last_layers(x, num_filters, out_filters, d3dim_filters, d3orient_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    d3dimy = compose(DarknetConv2D_BN_Leaky(num_filters*2, (3,3)), Dropout(0.25), DarknetConv2D(d3dim_filters, (1,1)))(x)

    d3orienty = compose(DarknetConv2D_BN_Leaky(num_filters*2, (3,3)), Dropout(0.25),
                       DarknetConv2D(d3orient_filters, (1,1)))(x)

    return x, y, d3dimy, d3orienty


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    print('yolo body =======')
    darknet = Model(inputs, darknet_body(inputs))
    x, y1, d3dimy1, d3orienty1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5), num_anchors*3, num_anchors*3*BIN)

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    #x = Concatenate()([x,darknet.layers[75].output])
  #  x = Concatenate()([x,darknet.layers[82].output])
    x, y2, d3dimy2, d3orienty2 = make_last_layers(x, 256, num_anchors*(num_classes+5), num_anchors*3, num_anchors*3*BIN)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    #x = Concatenate()([x,darknet.layers[57].output])
    #x = Concatenate()([x,darknet.layers[64].output])
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3, d3dimy3, d3orienty3 = make_last_layers(x, 128, num_anchors*(num_classes+5), num_anchors*3, num_anchors*3*BIN)

    return Model(inputs, [y1,y2,y3, d3dimy1, d3dimy2, d3dimy3, d3orienty1, d3orienty2, d3orienty3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    print('yolo body =======tiny ')
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

def yolo_19_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    print('yolo my yolo 19 =======tiny ')
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),                              #conv1 :16
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),   #maxpool1 
            DarknetConv2D_BN_Leaky(32, (3,3)),                              #conv2 :32
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),   #maxpool2     
            DarknetConv2D_BN_Leaky(64, (3,3)),                              #conv3 :64
            DarknetConv2D_BN_Leaky(32, (3,3)),                              #conv4 :32
            DarknetConv2D_BN_Leaky(64, (1,1)),                              #conv5 :64
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),   #maxpool3 
            DarknetConv2D_BN_Leaky(128, (3,3)),                             #conv6 :128
            DarknetConv2D_BN_Leaky(64, (1,1)),                              #conv7 :64
            DarknetConv2D_BN_Leaky(128, (3,3)),                             #conv8 :128
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),   #maxpool4
            DarknetConv2D_BN_Leaky(256, (3,3)),                             #conv9 :
            DarknetConv2D_BN_Leaky(128, (1,1)),                             #conv10 :
            DarknetConv2D_BN_Leaky(256, (3,3)),                             #conv12 :
            #DarknetConv2D_BN_Leaky(128, (3,3)),                             #conv13 :
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)                     #conv14 :

    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),   #maxpool5
          #  DarknetConv2D_BN_Leaky(512, (3,3)),                             #conv15
            DarknetConv2D_BN_Leaky(256, (3,3)),                             #conv16
          #  DarknetConv2D_BN_Leaky(512, (3,3)),                             #conv18
           # DarknetConv2D_BN_Leaky(256, (3,3)),                             #conv19
            DarknetConv2D_BN_Leaky(512, (3,3)),                             #conv20
         #   MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (1,1)))(x1)
           # DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
           # DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            DarknetConv2D_BN_Leaky(1024, (1,1)),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])



def yolo_head(d2feats, d3dimfeats, d3orientfeats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(d2feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(d2feats))

    d2feats = K.reshape(
        d2feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    d3dimfeats = K.reshape(d3dimfeats, [-1, grid_shape[0], grid_shape[1], num_anchors, 3])
    d3orientfeats = K.reshape(d3orientfeats, [-1, grid_shape[0], grid_shape[1], num_anchors, 3*BIN])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(d2feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(d2feats))
    box_wh = K.exp(d2feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(d2feats))
    box_confidence = K.sigmoid(d2feats[..., 4:5])
    box_class_probs = K.sigmoid(d2feats[..., 5:])
    # 3d dim noline 
    #box_dim = K.exp(feats[..., 5:8]) ##no exp 
    #box_dim = K.sigmoid(feats[..., 5:8]) ##no exp 
    box_dim = d3dimfeats[..., 0:3] ##no exp 
    box_3d_conf = K.sigmoid(d3orientfeats[..., 0:2])
    #print(box_3d_conf.shape)
    #box_3d_cossin = K.l2_normalize(K.reshape(d3orientfeats[..., 2:3*BIN], [-1, -1, -1, -1, BIN, 2]), 2)
    box_3d_cossin = d3orientfeats[..., 2:3*BIN]

    #print(box_3d_cossin.shape)
    #K.sigmoid(K.reshape(d3orientfeats[..., 2:3*BIN], [-1, BIN, 2]))
    #box_3d_cossin = K.l2_normalize(d3orientfeats[..., 2:3*BIN])
    #box_3d_cossin = K.tanh(d3orientfeats[..., 2:3*BIN])
    #K.sigmoid(feats[..., 8:14])

    if calc_loss == True:
        return grid, d2feats, box_xy, box_wh, box_dim, box_3d_conf, box_3d_cossin
    return box_xy, box_wh, box_confidence, box_dim, box_3d_conf, box_3d_cossin, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, d3dimfeats, d3orientfeats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_dim, box_3d_conf, box_3d_cossin, box_class_probs = yolo_head(feats, 
        d3dimfeats, d3orientfeats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    box_dim = K.reshape(box_dim, [-1, 3])
    box_3d_conf = K.reshape(box_3d_conf, [-1, 2])
    box_cossin = K.l2_normalize(K.reshape(box_3d_cossin, [-1, BIN, 2]), 2)
    #K.reshape(box_3d_cossin, [-1, 2, 2])
    return boxes, box_scores, box_dim, box_3d_conf, box_cossin


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs) // 3
    print(num_layers)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    box_dim = []
    box_3d_conf = []
    box_cossin = []
    for l in range(num_layers):
        _boxes, _box_scores, _box_dim, _box_3d_conf, _box_cossin = yolo_boxes_and_scores(yolo_outputs[l], yolo_outputs[num_layers+l], yolo_outputs[2*num_layers+l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
        box_dim.append(_box_dim)
        box_3d_conf.append(_box_3d_conf)
        box_cossin.append(_box_cossin)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    box_dim = K.concatenate(box_dim, axis=0)
    box_3d_conf = K.concatenate(box_3d_conf, axis=0)
    box_cossin = K.concatenate(box_cossin, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    box3ddim_ = []
    boxcossin_ = []
    box3dconf_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        box_dim = tf.boolean_mask(box_dim, mask[:, c])
        box_3d_conf = tf.boolean_mask(box_3d_conf, mask[:, c])
        box_cossin = tf.boolean_mask(box_cossin, mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        box_dim = K.gather(box_dim, nms_index)
        box_3d_conf = K.gather(box_3d_conf, nms_index)
        box_cossin = K.gather(box_cossin, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
        box3ddim_.append(box_dim)
        boxcossin_.append(box_cossin)
        box3dconf_.append(box_3d_conf)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    box3ddim_ = K.concatenate(box3ddim_, axis=0)
    box3dconf_ = K.concatenate(box3dconf_, axis=0)
    boxcossin_ = K.concatenate(boxcossin_, axis=0)

    return boxes_, scores_, classes_, box3ddim_, box3dconf_, boxcossin_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes+3+3*BIN),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5:8] = true_boxes[b,t,5:8] #/DIMSCALE #dim l, h, w
                    y_true[l][b, j, i, k, 8:14] = true_boxes[b,t,8:14] #8,9 conf  10 cos 11 sin 12 cos 13 sin
                    y_true[l][b, j, i, k, 14+c] = 1

    return y_true

def orientation_loss(y_true, y_pred, obj_mask, mf):
# Find number of anchors
    #print('orient loss ------')
    #print(test.shape)
    y_true = K.reshape(y_true*obj_mask, [-1, BIN, 2])
    y_pred = y_pred*obj_mask
    y_pred = K.l2_normalize(K.reshape(y_pred, [-1, BIN, 2]), 2)
    obj_mask = K.reshape(obj_mask, [-1, 1])
    #K.reshape(y_pred*obj_mask, [-1, BIN, 2])
    #anchors = K.sum(K.square(y_true), axis=2)
    #anchors = K.greater(anchors, tf.constant(0.5))
    #anchors = K.sum(K.cast(anchors, dtype='float32'), 1)
# Define the loss
# cos^2 + sin ^2 = 1
   # K.abs()
    #loss = K.abs(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    #print(tf.Session().run(y_true))
    #print(tf.Session().run(y_pred))
    #loss = K.switch(loss > 0.0, loss, K.zeros_like(loss))
    loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])   # -1 - 1
    loss = 1-loss
    loss = K.reshape(loss, [-1, 2])
    loss = loss*obj_mask

    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    losssum = K.sum(K.sum(loss, axis=0))
   # print(losssum.shape)
    allobj = K.sum(obj_mask)
    #print(allobj.shape)
  #  if K.eval(allobj) == 0:
  #      loss = 0.0
  #  else :
  #      loss = 4.0*(2 - 2 * (K.sum(K.sum(loss, axis=0))/allobj))
    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    #loss =  (allobj-K.sum(K.sum(loss, axis=0)))/mf
    #loss = tf.cond(allobj > 0, lambda: 3.0*(1 -  (K.sum(K.sum(loss, axis=0))/allobj)), lambda: 0.0)
    loss = tf.cond(allobj > 0, lambda: losssum/allobj, lambda: 0.0)

    #loss = 3.0 * K.abs(loss)
    #K.switch
    #loss = tf.cond(allobj > 0, lambda: (allobj-K.sum(K.sum(loss, axis=0)))/mf, lambda: 0.0)
    
    #loss = K.sum((2 - 2 * K.mean(loss,axis=0))) / anchors
    #print(loss.shape)
    return loss
    #K.mean(loss)

def orientation_loss3(y_true, y_pred, obj_mask, mf):
# Find number of anchors
    #print('orient loss ------')
    #print(test.shape)
    y_true = K.reshape(y_true*obj_mask, [-1, BIN, 2])
    y_pred = y_pred*obj_mask
    y_pred = K.l2_normalize(K.reshape(y_pred, [-1, BIN, 2]), 2)
    obj_mask = K.reshape(obj_mask, [-1, 1])
    #K.reshape(y_pred*obj_mask, [-1, BIN, 2])
    #anchors = K.sum(K.square(y_true), axis=2)
    #anchors = K.greater(anchors, tf.constant(0.5))
    #anchors = K.sum(K.cast(anchors, dtype='float32'), 1)
# Define the loss
# cos^2 + sin ^2 = 1
   # K.abs()
    #loss = K.abs(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    #print(tf.Session().run(y_true))
    #print(tf.Session().run(y_pred))
    #loss = K.switch(loss > 0.0, loss, K.zeros_like(loss))
    #loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])   # -1 - 1
    #loss = 1-loss
    cosd = K.square(y_true[:,:,0] - y_pred[:,:,0])
    sind = K.square(y_true[:,:,1] - y_pred[:,:,1])
    loss = cosd+sind
    #loss = K.reshape(loss, [-1, 2])
    loss = loss*obj_mask

    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    losssum = K.sum(K.sum(loss, axis=0))
   # print(losssum.shape)
    allobj = K.sum(obj_mask)
    #print(allobj.shape)
  #  if K.eval(allobj) == 0:
  #      loss = 0.0
  #  else :
  #      loss = 4.0*(2 - 2 * (K.sum(K.sum(loss, axis=0))/allobj))
    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    #loss =  (allobj-K.sum(K.sum(loss, axis=0)))/mf
    #loss = tf.cond(allobj > 0, lambda: 3.0*(1 -  (K.sum(K.sum(loss, axis=0))/allobj)), lambda: 0.0)
    loss = tf.cond(allobj > 0, lambda: losssum/allobj, lambda: 0.0)

    #loss = 3.0 * K.abs(loss)
    #K.switch
    #loss = tf.cond(allobj > 0, lambda: (allobj-K.sum(K.sum(loss, axis=0)))/mf, lambda: 0.0)
    
    #loss = K.sum((2 - 2 * K.mean(loss,axis=0))) / anchors
    #print(loss.shape)
    return loss
    #K.mean(loss)

def orientation_loss2(y_true, y_pred, obj_mask, mf):
# Find number of anchors
    #print('orient loss ------')
    #print(test.shape)
    #K.reshape(y_pred*obj_mask, [-1, BIN, 2])
    anchors = K.sum(K.square(y_true), axis=2)
    anchors = K.greater(anchors, tf.constant(0.5))
    anchors = K.sum(K.cast(anchors, dtype='float32'), 1)
# Define the loss
# cos^2 + sin ^2 = 1
   # K.abs()
    #loss = K.abs(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    #print(tf.Session().run(y_true))
    #print(tf.Session().run(y_pred))
    loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])   # -1 - 1
    #loss = K.switch(loss > 0.0, loss, K.zeros_like(loss))
    loss = 1-loss
    print(loss.shape)

    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    losssum = K.sum(K.sum(loss, axis=0))
   # print(losssum.shape)
    allobj = K.sum(obj_mask)
    #print(allobj.shape)
  #  if K.eval(allobj) == 0:
  #      loss = 0.0
  #  else :
  #      loss = 4.0*(2 - 2 * (K.sum(K.sum(loss, axis=0))/allobj))
    #loss = 4.0*K.sum((2 - 2 * K.mean(loss,axis=0)))
    #loss =  (allobj-K.sum(K.sum(loss, axis=0)))/mf
    #loss = tf.cond(allobj > 0, lambda: 3.0*(1 -  (K.sum(K.sum(loss, axis=0))/allobj)), lambda: 0.0)
    loss = tf.cond(allobj > 0, lambda: losssum/allobj, lambda: 0.0)

    #loss = 3.0 * K.abs(loss)
    #K.switch
    #loss = tf.cond(allobj > 0, lambda: (allobj-K.sum(K.sum(loss, axis=0)))/mf, lambda: 0.0)
    
    #loss = K.sum((2 - 2 * K.mean(loss,axis=0))) / anchors
    #print(loss.shape)
    return loss





def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou








def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=True):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers*3] #zzzzzzzzzzzzzzzzz
    print('list len')
    print(len(yolo_outputs))
    y_true = args[3*num_layers:]
    print(len(y_true))
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    torientloss = 0
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 14:]


        grid, raw_pred, pred_xy, pred_wh, box_dim, box_3d_conf, box_3d_cossin = yolo_head(yolo_outputs[l],
             yolo_outputs[num_layers + l], yolo_outputs[2*num_layers+l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

       # print('--------------')
       # print(pred_xy.shape)
       # print(box_3d_conf.shape)
       # print(box_3d_cossin.shape)
        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        raw_true_dim = y_true[l][..., 5:8]
        raw_true_3d_conf = y_true[l][..., 8:10]
        raw_true_3d_cossin = y_true[l][..., 10:14]
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        d3conf_loss = object_mask * K.binary_crossentropy(raw_true_3d_conf, box_3d_conf, from_logits=True)
        d3dim_loss = object_mask * K.square(raw_true_dim - box_dim)
        #print(box_3d_cossin.shape)
        
        #print(d3orient_loss.shape)
        #print('---loss-----')
        #print(object_mask.shape)
        #print(xy_loss.shape)
        #print(class_loss.shape)
        
        d3orient_loss = 2.0*orientation_loss3(raw_true_3d_cossin, box_3d_cossin, object_mask, mf) 
            
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        d3conf_loss = (K.sum(d3conf_loss) / mf)*1.0
        d3dim_loss = (K.sum(d3dim_loss) / mf)*2.0
        #d3orient_loss = d3orient_loss / mf
        #loss += xy_loss + wh_loss + confidence_loss + class_loss
        loss +=d3conf_loss + d3dim_loss + xy_loss + wh_loss + confidence_loss + class_loss + d3orient_loss 
        #loss += xy_loss + wh_loss + confidence_loss + class_loss 
       # + d3orient_loss
        #loss = d2loss+d3loss
        #tf.Print(d2loss, [d2loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='d2loss: ')
        #tf.Print(loss, [d3conf_loss, d3dim_loss, d3orient_loss], message='d3loss: ')
        torientloss += d3orient_loss

    if print_loss:
        loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, d3conf_loss, d3dim_loss, torientloss], message='loss: ')
            #loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask), d3conf_loss, d3dim_loss, d3orient_loss, mf], message='loss: ')
    return loss
