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
from tqdm import tqdm

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


def prune_dataset(annotation_file, path='/train2014/',num_examples=30000):
    PATH = path
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

    # selecting the first num_examples captions from the shuffled set
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return train_captions, img_name_vector, all_captions, all_img_name_vector

def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def text_one_hot(captions, k=5000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=k, 
                                                  oov_token="<unk>", 
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    seqs = tokenizer.texts_to_sequences(captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')
    # max_length = calc_max_length(seqs)
    return cap_vector

def cache_img_features(model, img_name_vector, batch_size=16):
    # getting the unique images
    encode_train = sorted(set(img_name_vector))

    # feel free to change the batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(
                                    encode_train).map(load_image).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = model(img)
        batch_features = tf.reshape(batch_features, 
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

def cache_doc2vec(model, captions):
    for caption in captions:
        pass


def get_tf_data(imgs, captions, doc2vec, batch_size=64, buffer_size=1000, parallel_workers=8, top_k=5000):

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                        oov_token="<unk>", 
                                                        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    seqs = tokenizer.texts_to_sequences(captions)
    tokenizer.word_index['<pad>'] = 0
    seqs = tokenizer.texts_to_sequences(captions)
    vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')
    # print(len(vector))
    doc2vec_captions = []
    for caption in captions:
        doc2vec_captions.append(doc2vec.transform(caption))
    doc2vec_captions = np.array(doc2vec_captions)
    dataset = tf.data.Dataset.from_tensor_slices((imgs, vector, doc2vec_captions))
    dataset = dataset.map(lambda item1, item2, item3: tf.py_func(
          map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.int32,  tf.float32]), num_parallel_calls=parallel_workers)
#     # shuffling and batching
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=tf.buffer_size)
    return dataset, tokenizer

def map_func(img_name, cap_vector, doc2vec_emb):
    img, _ = load_image(img_name)
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    # print(img, img_tensor, cap_vector, doc2vec_emb)
    return img, img_tensor, cap_vector, doc2vec_emb