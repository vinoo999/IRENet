from text_encoder import TextEncoder
from text_decoder import TextDecoder
from image_encoder import ImageEncoder
from image_decoder import ImageDecoder
import tensorflow as tf
import numpy as np
from tqdm import tqdm

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
