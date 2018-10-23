"""Miscellaneous utility functions."""

from functools import reduce

import cv2, os
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w*1.0/iw, h*1.0/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    print ('w %d h %d iw %d ih %d nw %d nh %d scale %f'%(w, h, iw, ih, nw, nh, scale))

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
#    image = image.resize(size, Image.BICUBIC)
    new_image.show()
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
VEHICLESNUM = [0, 1, 2, 3, 4, 5]
BIN, OVERLAP = 2, 0.1


def compute_anchors(angle):
    anchors = []
    
    wedge = 2.*np.pi/BIN
    l_index = int(angle/wedge)
    r_index = l_index + 1
    
    if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
        anchors.append([l_index, angle - l_index*wedge])
        
    if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
        anchors.append([r_index%BIN, angle - r_index*wedge])
        
    return anchors


def kitti_parse_annotation(label_dir, image_dir):
    image_num = 0
    all_image_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in VEHICLESNUM}
    dims_cnt = {key:0 for key in VEHICLESNUM}
    for label_file in sorted(os.listdir(label_dir)):
        all_objs = []
        image_file = label_file.replace('txt', 'png')
        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name': VEHICLES.index(line[0]),
                       'image':image_dir+image_file,
                       'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       'dims':np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]
                all_objs.append(obj)
        
        #print("objs len %d", len(all_objs))
        #print(all_objs)
        if len(all_objs)== 0:
            continue
        #print(all_objs)
        all_image_objs.append(all_objs)
        #all_objs.clear()
        image_num += 1

    #print(all_image_objs)
    ###### flip data
    for image_objs in all_image_objs:
        #print(len(image_objs))
        for obj in image_objs:
            # Fix dimensions
            obj['dims'] = obj['dims'] - dims_avg[obj['name']]

            # Fix orientation and confidence for no flip
            orientation = np.zeros((BIN,2))
            confidence = np.zeros(BIN)

            anchors = compute_anchors(obj['new_alpha'])

            for anchor in anchors:
                orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                confidence[anchor[0]] = 1.

            confidence = confidence / np.sum(confidence)

            obj['orient'] = orientation
            obj['conf'] = confidence

            # Fix orientation and confidence for flip
            orientation = np.zeros((BIN,2))
            confidence = np.zeros(BIN)

            anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
            for anchor in anchors:
                orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                confidence[anchor[0]] = 1
                
            confidence = confidence / np.sum(confidence)

            obj['orient_flipped'] = orientation
            obj['conf_flipped'] = confidence
            
    return all_image_objs


def get_random_data(image_objs, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    obj_cnt = 0
    h, w = input_shape
    #box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    
    box = np.zeros((len(image_objs), 5))   
    for obj in image_objs:
        if obj_cnt == 0:
           image = Image.open(obj['image'])
           iw, ih = image.size
        box[obj_cnt, 0] = obj['xmin']	
        box[obj_cnt, 1] = obj['ymin']	
        box[obj_cnt, 2] = obj['xmax']	
        box[obj_cnt, 3] = obj['ymax']	
        box[obj_cnt, 4] = obj['name']	
        obj_cnt += 1

    #print(box)
    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    #print('w %d h %d', iw, ih)
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
