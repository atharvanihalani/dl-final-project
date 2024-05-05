import tensorflow as tf
from model.encoder_decoder import TransformerEncoder, TransformerDecoder
from model.miscellaneous_layers import PositionalEncoding

## TODO: double-check the propagation of parametres/hyperparametres through
## the classes and layers

class MusicDecoderTransformer(tf.keras.Model): 

    def __init__(self, units, input_window_size, output_window_size, embed_size = 512, **kwargs):

        super().__init__()

        self.embed_size = embed_size
        # define model architecture here

        self.encoder_embedding = PositionalEncoding(embed_size, input_window_size)

        self.decoder_embedding = PositionalEncoding(embed_size, output_window_size)

        self.encoder = TransformerEncoder(
            units = 512,  
            num_encoder_blcks = 3, 
            num_heads = 3, 
            key_dim = 64, 
            ff_num_lyrs = 1,  
        )

        self.decoder = TransformerDecoder(
            units = 512,  
            num_decoder_blocks = 3, 
            num_heads = 3, 
            key_dim = 64, 
            ff_num_lyrs = 1,  
        )

        self.final_dense = tf.keras.layers.Dense(units, activation = "sigmoid")

    def call(self, inputs): 

        context, x = inputs

        context = self.encoder_embedding(context)

        encoder_out = self.encoder(context) 

        x = self.decoder_embedding(x)

        decoder_out = self.decoder(x, encoder_out)

        piano_roll_vector = self.final_dense(decoder_out)

        return piano_roll_vector

