import numpy as np
import tensorflow as tf

class FeedForward(tf.keras.layers.Layer): 

    def __init__(self, num_layers, units_end, hidden_size=None, **kwargs): 

        super().__init__(**kwargs)

        self.units_end = units_end

        if not hidden_size: 
            self.hidden_size = units_end
        else: self.hidden_size = hidden_size

        layer_list = [tf.keras.layers.Dense(hidden_size, activation = "relu") for _ in range(num_layers-1)]
        layer_list.append(tf.keras.layers.Dense(units_end, activation = "relu"))

        self.dense = tf.keras.Sequential(layer_list)

        self.normalise = tf.keras.layers.LayerNormalization()

    def call(self, x): 

        out = self.dense(x)
        out = self.normalise(out)

        return out
    
def positional_encoding(length, depth):
    """Copied from Siddharta Laloux's HW5p"""
    ## Can remove signature
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """Copied from Siddharta Laloux's HW5p"""
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ##TODO: Embed labels into an optimizable embedding space
        ## NOTE: no vocab defined, this is probably a Dense layer or sequence 
        ## of dense layers
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)

    def call(self, x):
        ## Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        # length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding
        return x
    
