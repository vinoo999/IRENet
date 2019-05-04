import tensorflow as tf

def InceptionWrapper(init_weights='imagenet'):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights=init_weights)
    new_input = image_model.input
    outputs = image_model.layers[-1].output
    model = tf.keras.Model(new_input, outputs)
    return model

class ImageEncoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(2*embedding_dim)
        self.fc2 = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        return x


