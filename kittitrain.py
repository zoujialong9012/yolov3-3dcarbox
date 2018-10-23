"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from kitti_data_processing import get_random_data, kitti_parse_annotation 
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/kitti_classes.txt'
     #voc_classes.txt
    anchors_path = 'model_data/yolo_anchors.txt'
    #anchors_path = 'model_data/kitti_yolo_anchors.txt'
    label_dir = '/home/cidi/dl/3dobject/3D-Deepbox/training/label_2/'
    image_dir = '/home/cidi/dl/3dobject/3D-Deepbox/training/image_2/'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

#    input_shape = (416,416) # multiple of 32, hw
    input_shape = (384, 960) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, load_pretrained=True, weights_path='model_data/yolo_weights.h5')
            #'logs/000/trained_weights_stage_1.h5')#'model_data/yolo_weights.h5') # make sure you know what you freeze
    
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

#    with open(annotation_path) as f:
#        lines = f.readlines()
#    np.random.seed(10101)
#    np.random.shuffle(lines)
#    np.random.seed(None)
#    num_val = int(len(lines)*val_split)
#    num_train = len(lines) - num_val
#    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#    sess = tf.Session(config=config)
    #if GPU:
    #   num_GPU = 1
    #   num_CPU = 1
    #if CPU:
#    num_CPU = 4
#    num_GPU = 0
#
#    config = tf.ConfigProto(intra_op_parallelism_threads=1,\
#        inter_op_parallelism_threads=1, allow_soft_placement=True,\
#        device_count = {"CPU" : num_CPU, "GPU" : num_GPU})
#    session = tf.Session(config=config)
#    K.set_session(session)	
#

    val_split = 0.1
    all_image_objs = kitti_parse_annotation(label_dir, image_dir)
#    print("all image len %d", len(all_image_objs))
#    print(all_image_objs)
    np.random.seed(10101)
    np.random.shuffle(all_image_objs)
    num_val = int(len(all_image_objs) * val_split)
    num_train = len(all_image_objs) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 6
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(all_image_objs[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(all_image_objs[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=20,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_20.h5')

    if True:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 6
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(all_image_objs[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(all_image_objs[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=20,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_50.h5')

    if True:
        model.compile(optimizer=Adam(lr=1e-5), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 6
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(all_image_objs[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(all_image_objs[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=70,
                initial_epoch=50,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_70.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-5), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 6 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(all_image_objs[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(all_image_objs[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=70,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    for layer in model_body.layers:#range(len(model_body.layers)-1):
        print(layer)

    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-80)[freeze_body-1]
            #for i in range(num): model_body.layers[i].trainable = False
           # for i in range(num):
            #    model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
def data_generator(all_image_objs, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(all_image_objs)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(all_image_objs)
            imaged, image, box, picname = get_random_data(all_image_objs[i], input_shape, random=True)
        #    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * 416 + 0.5).astype('int32'))
        #    draw = ImageDraw.Draw(imaged)
        #    for box1 in box:
        #        #print('box'+str(box1[4]))
        #        #print(box1)
        #        left, top, right, bottom, c = box1
        #        label = '{}'.format(VEHICLES[int(c)])
        #        label_size = draw.textsize(label, font)
        #        if top - label_size[1] >= 0:
        #            text_origin = np.array([left, top - label_size[1]])
        #        else:
        #            text_origin = np.array([left, top + 1])

        #        draw.rectangle([left, top, right, bottom],
        #            outline=(255, 0, 0))
        #        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
        #                fill=(0, 255, 0))
        #        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #    del draw
        #    #imaged.show(title=picname)
        #    plt.title(picname)
        #    plt.imshow(imaged)
        #    plt.show()

            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(all_image_objs, batch_size, input_shape, anchors, num_classes):
    n = len(all_image_objs)
    #print("len %d"%(n))
    if n==0 or batch_size<=0: return None
    return data_generator(all_image_objs, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
