import tensorflow as tf
from attention import CrossAttentionResidual, SelfAttentionResidual, CausalSelfAttentionResidual
from miscellaneous_layers import FeedForward

class EncoderBlock(tf.keras.layers.Layer): 

    def __init__(self, num_heads, key_dim, ff_num_lyrs, ff_units, ff_hidden = None, **kwargs): 

        super().__init__(**kwargs)

        self.self_attention = SelfAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.ffn = FeedForward(ff_num_lyrs, ff_units, ff_hidden)


    def call(self, x): 

        out = self.self_attention(x)
        out = self.ffn(out)

        return out
    
class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_encoder_blcks, num_heads, key_dim ,**kwargs): 

        super().__init__()

        self.num_encoder_blcks = num_encoder_blcks
        self.num_heads = num_heads
        self.key_dim = key_dim

        encoder_list = [
            EncoderBlock(num_heads, key_dim, **kwargs) for _ in range(num_encoder_blcks)
        ]

        self.encoders = tf.keras.Sequential(encoder_list)

    def call(self, x):
        
        return self.encoders(x)


test_model = TransformerEncoder(4, 6, 3)
print("hello!")


class DecoderBlock(tf.keras.layers.Layer): 

    def __init__(self, num_heads, key_dim, ff_num_lyrs, ff_units, ff_hidden = None, **kwargs): 

        super().__init__()

        self.cross_attention = CrossAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.causal_self_atten = CausalSelfAttentionResidual(
            num_heads = num_heads,
            key_dim = key_dim
        )

        self.ffn = FeedForward(ff_num_lyrs, ff_units, ff_hidden)
        
