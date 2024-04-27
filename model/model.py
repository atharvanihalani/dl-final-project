class MusicDecoderTransformer(tf.keras.Model): 

    def __init__(self, embed_size, **kwargs):

        super.__init__(**kwargs)

        self.embed_size = embed_size
        # define model architecture here

        self.embeddings = tf.keras.layers.Dense(units = self.embed_size)

        self.encoder = tf.nlp.models.TransformerEncoder(
            num_layers = 8, 
            num_attention_heads = 6, 
            intermediate_size = 1024, 
        )

        self.decoder = tf.nlp.models.TransformerDecoder(

        )

