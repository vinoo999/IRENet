import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WordEncoder(object):
    def __init__(self, embedding_size=128):
        self.stop_words = stopwords.words('english')
        self.EMBEDDING_SIZE = embedding_size
        self.model = self.train()

    # Arguments
    # caption: string
    #       example: "<start> A skateboarder performing a trick on a skateboard ramp. <end>"
    # returns: vector
    #       example: [-2.33201645e-04  1.73898158e-03  1.91628770e-03  1.21335301e-03 ... ]
    def transform(self, caption):
        tokenized_words = caption.split(" ")
        x = [word for word in tokenized_words if word.lower() not in self.stop_words]
        return model.infer_vector(x)
    
    def train(self):
        documents = []
        _, all_captions = pickle.load( open( "captions.pickle", "rb" ) )
        for i, caption in enumerate(all_captions):
            tokenized_words = caption.split(" ")
            x = [word for word in tokenized_words if word.lower() not in self.stop_words]
            # TODO: should I remove <start> and <end>
            documents.append(TaggedDocument(x, [i]))
        # TODO: optimize parameters of model
        model = Doc2Vec(documents, vector_size=self.EMBEDDING_SIZE, workers=4)
