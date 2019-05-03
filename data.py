import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob 
from PIL import Image
import pickle

def download(path='captions.zip'):
    annotation_zip = tf.keras.utils.get_file(path, 
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip, 
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                            extract = True)
        PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
        PATH = os.path.abspath('.')+'/train2014/'

    return annotation_file, PATH


def prune_dataset(annotation_file, image_dir):
    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
        
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffling the captions and image_names together
    # setting a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

    # selecting the first 30000 captions from the shuffled set
    num_examples = 30000
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]