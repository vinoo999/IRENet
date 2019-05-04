import tensorflow as tf

class ImageDecoder(tf.keras.Model):
    # assumes 299,299,3 output size.
    def __init__(self, embedding_dim):
        super(ImageDecoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.decode_fc1 = tf.keras.layers.Dense(embedding_dim, activation='relu', name='decode_fc1')
        self.decode_fc2 = tf.keras.layers.Dense(2048, activation='relu', name='decode_fc2')
        self.reshape = tf.keras.layers.Reshape((8,8,2048))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2), name='deconv1')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2), name='deconv2')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5,5), activation='relu', padding='valid', strides=(2,2), name='deconv3')
        self.deconv4 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(4,4), activation='relu', padding='valid', strides=(2,2), name='deconv4')
        self.deconv5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5,5), activation='sigmoid', padding='valid', strides=(2,2), name='deconv5')

        self.dropout = tf.keras.layers.Dropout(0.2)
        # self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.decode_fc1(x)
        if training:
            x = self.dropout(x)
        x = self.decode_fc2(x)
        if training:
            x = self.dropout(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        # x = self.batch_norm(x)
        x = self.deconv2(x)
        # x = self.batch_norm(x)
        x = self.deconv3(x)
        # x = self.batch_norm(x)
        x = self.deconv4(x)
        # x = self.batch_norm(x)
        x = self.deconv5(x)

        return x