from text_encoder import TextEncoder
from text_decoder import TextDecoder
from image_encoder import ImageEncoder
from image_decoder import ImageDecoder
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import data

class IreNet(object):
    def __init__(self, *args):
        self._bulid_arch()
        pass
    
    def _build_arch(self):
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.image_input = tf.placeholder()
        return

    def call(self, x):

        return
    
def cache_img_features(model, img_name_vector, batch_size=16):
    # getting the unique images
    encode_train = sorted(set(img_name_vector))

    # feel free to change the batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(
                                    encode_train).map(data.load_image).batch(16)

    for img, path in image_dataset:
        batch_features = model(img)
        batch_features = tf.reshape(batch_features, 
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())