import tensorflow as tf
from encoder_decoder import TransformerEncoder, TransformerDecoder
from miscellaneous_layers import PositionalEncoding

## TODO: double-check the propagation of parametres/hyperparametres through
## the classes and layers

class MusicDecoderTransformer(tf.keras.Model): 

    def __init__(self, units, window_size, embed_size = 512, **kwargs):

        super().__init__()

        self.embed_size = embed_size
        # define model architecture here

        self.embedding = PositionalEncoding(embed_size, window_size)

        self.encoder = TransformerEncoder(
            units = 1024,  
            num_encoder_blcks = 8, 
            num_heads = 6, 
            key_dim = 64, 
            ff_num_lyrs = 1,  
        )

        self.decoder = TransformerDecoder(
            units = 1024,  
            num_decoder_blcks = 8, 
            num_heads = 6, 
            key_dim = 64, 
            ff_num_lyrs = 1,  
        )

        self.final_dense = tf.keras.layers.Dense(units, activation = "sigmoid")

    def call(self, x): 

        context, x = self.embedding(x)

        encoder_out = self.encoder(context) 

        decoder_out = self.decoder(x, encoder_out)

        piano_roll_vector = self.final_dense(decoder_out)

        return piano_roll_vector

