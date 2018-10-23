#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

from multiprocessing import Process
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

import cv2, os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

#learning_rate = 0.0006
learning_rate = 0.0001
BIN, OVERLAP = 2, 0.1
#### Placeholder
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
o_label = tf.placeholder(tf.float32, shape = [None, BIN, 2])
c_label = tf.placeholder(tf.float32, shape = [None, BIN])



class YOLO(object):
    def __init__(self):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
    
        #graph=tf.Graph()
        self.d3sess = tf.Session(config=tfconfig) #, graph=graph)
        init = tf.global_variables_initializer()
        self.d3sess.run(init)
        #with sess.as_default():
        #    with graph.as_default():
        self.d3dimension, self.d3orientation, self.d3confidence, self.d3loss, self.d3optimizer, self.d3loss_d, self.d3loss_o, self.d3loss_c = build_model()
        saver = tf.train.Saver()
        model_file=tf.train.latest_checkpoint('/home/cidi/dl/3dobject/3D-Deepbox/model')
        saver.restore(self.d3sess, model_file)
 
#        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        #self.model_path = 'logs/000/trained_weights_final.h5' # model path or trained weights path
        #self.model_path = 'logs/000/trained_weights_stage_1.h5' # model path or trained weights path
        self.model_path = 'logs/000/trained_weights_stage_70back.h5' # model path or trained weights path
    #    self.model_path3d = '/home/cidi/dl/3dobject/3D-Deepbox/model'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        #self.classes_path = 'model_data/coco_classes.txt'
        self.classes_path = 'model_data/kitti_classes.txt'
        self.graph = tf.Graph()
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        with self.graph.as_default():
            print('tf session tf graph')
            self.sess = tf.Session(graph = self.graph)#K.get_session()
    #    self.sess = K.get_session()
        #K.set_session(self.sess)
        #self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.model_image_size = (960, 384) # fixed size or (None, None), hw
        with self.sess.as_default():
            with self.graph.as_default():
                self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        #self.yolo_model = load_model(model_path, compile=False)
        try:
            print('load model session')
            print(model_path)
            #with self.sess.as_default():
                #print('==============================>load model sess')
            self.yolo_model = load_model(model_path, compile=False)
        except:
        #    print('tinnnnnnnnyyyyyyyyyyyyyyyy')
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        with self.sess.as_default():
            with self.graph.as_default():
                if self.model_image_size != (None, None):
                    assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
                    assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
                    boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
                else:
                    new_image_size = (image.width - (image.width % 32),
                                      image.height - (image.height % 32))
                    boxed_image = letterbox_image(image, new_image_size)
                image_data = np.array(boxed_image, dtype='float32')

                print(image_data.shape)
                image_data /= 255.
                image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

                out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        self.yolo_model.input: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })

                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

                font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300
                o_classes = []
                for i, c in reversed(list(enumerate(out_classes))):
                    predicted_class = self.class_names[c]
                    box = out_boxes[i]
                    score = out_scores[i]
                    o_classes.append(predicted_class)

                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=self.colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw

                end = timer()
                print(end - start)
        return image, out_boxes, o_classes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        #image = yolo.detect_image(image)
        image = detect_3d_box(yolo, image, frame)#'007517.png')
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def build_model():

  #### build some layer 
  def LeakyReLU(x, alpha):
      return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

  def orientation_loss(y_true, y_pred):
      # Find number of anchors
      anchors = tf.reduce_sum(tf.square(y_true), axis=2)
      anchors = tf.greater(anchors, tf.constant(0.5))
      anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

      # Define the loss
      loss = (y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
      loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss,axis=0))) / anchors

      return tf.reduce_mean(loss)

  #####
  #Build Graph
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    conv5 = tf.contrib.layers.flatten(net)

    #dimension = slim.fully_connected(conv5, 512, scope='fc7_d')
    dimension = slim.fully_connected(conv5, 512, activation_fn=None, scope='fc7_d')
    dimension = LeakyReLU(dimension, 0.1)
    dimension = slim.dropout(dimension, 0.5, scope='dropout7_d')
    #dimension = slim.fully_connected(dimension, 3, scope='fc8_d')
    dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc8_d')
    #dimension = LeakyReLU(dimension, 0.1)

    #loss_d = tf.reduce_mean(tf.square(d_label - dimension))
    loss_d = tf.losses.mean_squared_error(d_label, dimension)

    #orientation = slim.fully_connected(conv5, 256, scope='fc7_o')
    orientation = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_o')
    orientation = LeakyReLU(orientation, 0.1)
    orientation = slim.dropout(orientation, 0.5, scope='dropout7_o')
    #orientation = slim.fully_connected(orientation, BIN*2, scope='fc8_o')
    orientation = slim.fully_connected(orientation, BIN*2, activation_fn=None, scope='fc8_o')
    #orientation = LeakyReLU(orientation, 0.1)
    orientation = tf.reshape(orientation, [-1, BIN, 2])
    orientation = tf.nn.l2_normalize(orientation, dim=2)
    loss_o = orientation_loss(o_label, orientation)

    #confidence = slim.fully_connected(conv5, 256, scope='fc7_c')
    confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_c')
    confidence = LeakyReLU(confidence, 0.1)
    confidence = slim.dropout(confidence, 0.5, scope='dropout7_c')
    confidence = slim.fully_connected(confidence, BIN, activation_fn=None, scope='fc8_c')
    loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= confidence))
   
    confidence = tf.nn.softmax(confidence)
    #loss_c = tf.reduce_mean(tf.square(c_label - confidence))
    #loss_c = tf.losses.mean_squared_error(c_label, confidence)
    
 #   print("lossd: ", loss_d, "losso: ", loss_o)
    total_loss = 4. * loss_d + 8. * loss_o + loss_c
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
     #https://www.jianshu.com/p/e6e8aa3169ca
#    optimizer = tf.train.AdamOptimizer().minimize(total_loss)
#    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
    
    return dimension, orientation, confidence, total_loss, optimizer, loss_d, loss_o, loss_c



NORM_H, NORM_W = 224, 224
dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}

VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
import matplotlib.pyplot as plt
def detect_3d_box(yolo, image, frame):
    outimage, out_boxs, out_classes = yolo.detect_image(image)    
    #outimage.show()

    oimage = frame
    ih = oimage.shape[0]
    iw = oimage.shape[1]
    print(out_boxs)
    for i in range(len(out_boxs)):  
        cls_name = out_classes[i]
        box = out_boxs[i]
        #print('w %d h %d 0 %f 1 %f 2 %f 3 %f'%(iw, ih, box[0], box[1], box[2], box[3]))
        if (box[0] > ih or box[0] < 0 or box[1] > iw or box[1] < 0 or box[2] > ih or box[2] < 0 or box[3] > iw or box[3] < 0):
            continue
        patch = oimage[np.floor(box[0] + 0.5).astype('int32'):np.floor(box[2] + 0.5).astype('int32'), np.floor(box[1] + 0.5).astype('int32'):np.floor(box[3] + 0.5).astype('int32')]
        patch = cv2.resize(patch, (NORM_H, NORM_W))
       # print(patch.shape)
        #cv.ShowImage("roi", patch)
        #cv2.imshow("image", patch)
        #cv2.waitKey(20)
        #plt.imshow(patch),plt.title('Image')
        #plt.show()
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])   # - rgb avg
    #    print(patch.shape)
        patch = np.expand_dims(patch, 0)
    #    print(patch.shape)
        with yolo.d3sess.as_default():
            with yolo.d3sess.graph.as_default():
                prediction = yolo.d3sess.run([yolo.d3dimension, yolo.d3orientation, yolo.d3confidence], feed_dict={inputs: patch})
       # print(prediction)
        # Transform regressed angle
        max_anc = np.argmax(prediction[2][0])
        anchors = prediction[1][0][max_anc]

        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])

        wedge = 2.*np.pi/BIN
        angle_offset = angle_offset + max_anc*wedge
        angle_offset = angle_offset % (2.*np.pi)

        angle_offset = angle_offset - np.pi/2
        if angle_offset > np.pi:
            angle_offset = angle_offset - (2.*np.pi)

        #diffa = float(line[3]) - angle_offset
        #line[3] = str(angle_offset)
        #line[-1] = angle_offset +np.arctan(float(line[11]) / float(line[13]))
        
        # Transform regressed dimension
        if cls_name in VEHICLES:
            dims = dims_avg[cls_name] + prediction[0][0]
        else:
            dims = dims_avg['Car'] + prediction[0][0]

        #print(prediction[0][0])
        print('h %f w %f l %f'%(dims[0], dims[1], dims[2]))
        line = list(dims)
        print (line)
        print(np.max(prediction[2][0]))
        print('angle %f'%(angle_offset))

    return outimage

#def yolov3_proc(yolo,l):



        
class yoloProcess():# (Process):
    def __init__(self, imagename):
    #    super().__init__()
        self.yolo = YOLO()
        self.image = imagename
        
    def run(self):
        image = Image.open(self.image)
        # oimage = oimage.astype(np.float32, copy=False)
        outimage, out_boxs, out_classes = self.yolo.detect_image(image)    
        outimage.show()
        print(out_boxs)



if __name__ == '__main__':
    #detect_img(YOLO())
    #yolop = yoloProcess('000008.png')
    #yolop.run()
#    detect_3d_box(YOLO(), 'testout', '000008.png')
    detect_video(YOLO(), '1.avi', output_path="")

