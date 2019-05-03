import tensorflow as tf

def ImageEncoder(init_weights='imagenet'):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights=init_weights)
    new_input = image_model.input
    outputs = image_model.output
    model = tf.keras.Model(new_input, outputs)
    return model
