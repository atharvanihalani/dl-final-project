import tensorflow as tf
from model.attention import CrossAttentionResidual, SelfAttentionResidual, CausalSelfAttentionResidual
from model.miscellaneous_layers import FeedForward

class EncoderBlock(tf.keras.layers.Layer): 

    def __init__(self, num_heads, key_dim, units, ff_num_lyrs, ff_hidden = None, **kwargs): 

        super().__init__(**kwargs)

        self.self_attention = SelfAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.ffn = FeedForward(ff_num_lyrs, units, ff_hidden)


    def call(self, x): 

        out = self.self_attention(x)
        out = self.ffn(out)

        return out
    
class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, units, num_encoder_blcks, num_heads, key_dim ,**kwargs): 

        super().__init__()

        self.num_encoder_blcks = num_encoder_blcks
        self.num_heads = num_heads
        self.key_dim = key_dim

        encoder_list = [
            EncoderBlock(num_heads, key_dim, units, **kwargs) for _ in range(num_encoder_blcks)
        ]

        self.encoders = tf.keras.Sequential(encoder_list)

    def call(self, x):
        
        return self.encoders(x)


class DecoderBlock(tf.keras.layers.Layer): 

    def __init__(self, num_heads, key_dim, units, ff_num_lyrs, ff_hidden = None, **kwargs): 

        super().__init__()

        self.cross_attention = CrossAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.causal_self_atten = CausalSelfAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.ffn = FeedForward(ff_num_lyrs, units, ff_hidden)
        
    def call(self, x, encoder_out): 
        
        out = self.causal_self_atten(x)
        out = self.cross_attention(x, encoder_out)

        out = self.ffn(out)

        return out
    
class TransformerDecoder(tf.keras.layers.Layer): 

    def __init__(self, units, num_decoder_blocks, num_heads, key_dim, **kwargs): 

        super().__init__()

        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.decoder_list = [
            DecoderBlock(num_heads, key_dim, units, **kwargs) for _ in range(num_decoder_blocks)
        ]

    def call(self, x, encoder_out): 

        out = x

        for i in range(self.num_decoder_blocks): 
            out = self.decoder_list[i](out, encoder_out) 

        return out