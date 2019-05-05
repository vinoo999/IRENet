from text_encoder import TextEncoder
from text_encoder import Doc2VecEncoder
from text_decoder import TextDecoder 
from image_encoder import ImageEncoder
from image_encoder import InceptionWrapper
from image_decoder import ImageDecoder
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import data
import time

class IreNet(object):
    def __init__(self, embedding_dim=512, max_vocab_size=5000):  
        self.embedding_dim = 512
        self.units = 512
        self.vocab_size = max_vocab_size
        self._build_arch(max_vocab_size)
    
    def _build_arch(self, max_vocab_size=5000):
        self.inception = InceptionWrapper()
        self.doc2vec = Doc2VecEncoder(embedding_size=4096, caption_pickle='captions.pickle', max_vocab_size=max_vocab_size)
        self.text_encoder = TextEncoder(self.embedding_dim)
        self.image_encoder = ImageEncoder(self.embedding_dim)

        self.text_decoder = TextDecoder(self.embedding_dim, self.units, self.vocab_size)
        self.image_decoder = ImageDecoder(self.embedding_dim)
        return

    def rnn_loss(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_, name='rnn_loss')
    
    def img_loss(self, real, pred):
        return tf.nn.l2_loss(real - pred, name='reconstruction_loss')

    def latent_loss(self, img_emb, text_emb):
        return tf.nn.l2_loss(img_emb - text_emb, name='latent_loss')
    

def train(model, dataset, tokenizer, batch_size=16, epochs=20):
    loss_plot = []
    EPOCHS = epochs
    optimizer = tf.train.AdamOptimizer()
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        
        for (batch, (img, img_tensor, target, doc2vec_emb)) in enumerate(dataset):
            print("Batch {} in time {}".format(batch+1, time.time()-start))
            loss = 0
            
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = model.text_decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)
            
            with tf.GradientTape() as tape:
                features = model.image_encoder(img_tensor)
                print("Image Encoder in time {}".format(time.time()-start))
                features2 = model.text_encoder(doc2vec_emb)
                print("Doc2Vec Encoder in time {}".format(time.time()-start))

                feature_shape = features.shape
                feature_size = np.float32(1.0)
                for dim in feature_shape:
                    feature_size*=int(dim)
                
                latent_loss = model.latent_loss(features, features2) / feature_size
            
                features_out = np.mean([features, features2], axis=0)
                print("loss calc in time {}, {}".format(time.time()-start, latent_loss))
                img_size = np.float32(1.0)
                img_shape = img.shape
                for dim in img_shape:
                    img_size*=int(dim)
                recon = model.image_decoder(features_out)
                recon_loss = model.img_loss(img, recon) / img_size

                print("recon in time {}, {}".format(time.time()-start, recon_loss))
                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = model.text_decoder(dec_input, features_out, hidden)

                    loss += model.rnn_loss(target[:, i], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)
            
                text_loss = loss / int(target.shape[1])
                final_loss = text_loss + latent_loss + recon_loss
            
            print("RNN in time {} {}".format(time.time()-start, text_loss))
            total_loss += final_loss
            
            variables = model.image_encoder.variables + model.text_encoder.variables + model.text_decoder.variables + model.image_decoder.variables
            # print(variables)
            gradients = tape.gradient(final_loss, variables) 
            print("Gradient in time {}".format(time.time()-start))
            # print(gradients)
            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
            print("Apply grad in time {}".format(time.time()-start))
            
            if batch % 1 == 0:
                print ('Epoch {} Batch {} Latent Loss {:.4f} Recon Loss {:.4f} Text Loss {:.4f} Total Loss {:.4f}'.format(epoch + 1, 
                                                            batch, 
                                                            latent_loss.numpy(), recon_loss.numpy(), text_loss.numpy(), final_loss.numpy()))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss)
        
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                            total_loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    pass