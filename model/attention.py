import tensorflow as tf


class MHAttentionResidual(tf.keras.layers.Layer): 
    """
    Class for general attention. TREAT AS AN ABSTRACT CLASS

    Attributes: 
    
    multi_head_attention: tf.keras.layers.MultiHeadAttention()
        A multi-headed attention layer

    normalise: tf.keras.layers.LayerNormalization()
        Layer normalisation

    self.add: tf.keras.layers.Add()
        Residual connection

    Functions: 
        __init()__
            to initialise the class
    """
    
    def __init__(self, **kwargs): 
        
        super().__init__()

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.normalise = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttentionResidual(MHAttentionResidual): 

    def __init__(self, **kwargs): 

        super().__init__(**kwargs)

    def call(self, x, context): 

        out = self.multi_head_attention(
            query = x, 
            key = context, 
            value = context,
            return_attention_scores = False, 
        )

        residual = self.add([x, out])
        x = self.normalise(residual)

        return x
    
class SelfAttentionResidual(MHAttentionResidual): 

    def __init__(self, **kwargs): 

        super().__init__(**kwargs)

    def call(self, x): 

        out = self.multi_head_attention(
            query = x, 
            key = x, 
            value = x,
            return_attention_scores = False, 
        )

        residual = self.add([x, out])
        x = self.normalise(residual)

        return x
    
class CausalSelfAttentionResidual(MHAttentionResidual): 
    """
    Class for causal attention (i.e. future context is masked). Likely not 
    needed for this task but interesting if we want to play around with music 
    generation
    """

    def __init__(self, **kwargs): 

        super().__init__(**kwargs)

    def call(self, x): 

        out = self.multi_head_attention(
            query = x, 
            key = x, 
            value = x,
            use_causal_mask = True, 
            return_attention_scores = False, 
        )

        residual = self.add([x, out])
        x = self.normalise(residual)

        return x
    
