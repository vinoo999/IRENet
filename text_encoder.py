import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tensorflow as tf

class Doc2VecEncoder(object):
    def __init__(self, embedding_size=128, caption_pickle='captions.pickle', max_vocab_size=None):
        self.stop_words = stopwords.words('english')
        self.EMBEDDING_SIZE = embedding_size
        self.train(caption_pickle, max_vocab_size)

    # Arguments
    # caption: string
    #       example: "<start> A skateboarder performing a trick on a skateboard ramp. <end>"
    # returns: vector
    #       example: [-2.33201645e-04  1.73898158e-03  1.91628770e-03  1.21335301e-03 ... ]
    def transform(self, caption):
        tokenized_words = caption.split(" ")
        x = [word for word in tokenized_words if word.lower() not in self.stop_words]
        return self.model.infer_vector(x)
    
    def train(self, caption_pickle="captions.pickle", max_vocab_size=None):
        documents = []
        _, all_captions= pickle.load( open( caption_pickle, "rb" ) )
        for i, caption in enumerate(all_captions):
            tokenized_words = caption.split(" ")
            x = [word for word in tokenized_words if word.lower() not in self.stop_words]
            # TODO: should I remove <start> and <end>
            documents.append(TaggedDocument(x, [i]))
        # TODO: optimize parameters of model
        self.model = Doc2Vec(documents, vector_size=self.EMBEDDING_SIZE, workers=4, max_vocab_size=max_vocab_size)

class TextEncoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(TextEncoder, self).__init__()
        self.encode_fc1 = tf.keras.layers.Dense(64*64, activation='relu', name='encode_fc1')
        self.encode_fc2 = tf.keras.layers.Dense(64*64, activation='relu', name='encode_fc2')
        self.reshape = tf.keras.layers.Reshape((64,64))
        self.encode_fc3 = tf.keras.layers.Dense(embedding_dim, activation='relu', name='encode_fc3')

    def call(self, x):
        x = self.encode_fc1(x)
        x = self.encode_fc2(x)
        x = self.reshape(x)
        x = self.encode_fc3(x)
        return x

