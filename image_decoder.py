import tensorflow as tf

class ImageDecoder(tf.keras.Model):
    # assumes 299,299,3 output size.
    def __init__(self, embedding_dim):
        super(ImageDecoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.reshape = tf.keras.layers.Reshape((8,8,embedding_dim))
        self.deconv1 = tf.keras.layers.Conv2D(128, (3,3), padding='same')# 8x8
        self.upsample1 = tf.keras.layers.UpSampling2D((3,3))# 24x24
        self.deconv2 = tf.keras.layers.Conv2D(64, (5,5), padding='valid') # 20x20
        self.upsample2 = tf.keras.layers.UpSampling2D((3,3)) # 60x60
        self.deconv3 = tf.keras.layers.Conv2D(16, (5,5), padding='same') # 56x56
        self.upsample3 = tf.keras.layers.UpSampling2D((3,3)) # 168x168
        self.deconv4 = tf.keras.layers.Conv2D(3, (5,5), padding='same') # 164x164
        self.upsample4 = tf.keras.layers.UpSampling2D((2,2)) # 168x168
        self.crop = tf.keras.layers.Cropping2D(cropping=((15, 15), (14, 14)))

    def call(self, x):
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.upsample1(x)
        # x = self.batch_norm(x)
        x = self.deconv2(x)
        x = self.upsample2(x)
        # x = self.batch_norm(x)
        x = self.deconv3(x)
        x = self.upsample3(x)
        x = self.deconv4(x)
        x = self.upsample4(x)
        x = self.crop(x)
        return x