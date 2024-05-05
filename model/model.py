import tensorflow as tf
from model.music_transformer import MusicDecoderTransformer
import os

def get_compiled_model():
     
    # music_transformer = MusicDecoderTransformer(
    #     units = 88,
    #     input_window_size = 217, 
    #     output_window_size = 110249, 
    #     embed_size = 512,
    # ) 

    music_transformer = MusicDecoderTransformer(
        units = 88,
        input_window_size = 43, 
        output_window_size = 22049, 
        embed_size = 512,
    ) 

    music_transformer.compile(
        optimizer="adam", 
        loss = tf.keras.losses.KLDivergence(
            reduction='sum_over_batch_size', name='kl_divergence'
        ), # loss = tf.keras.losses.BinaryCrossentropy(from_logits = False), 
        metrics = [
            tf.keras.metrics.BinaryAccuracy(), 
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall(), 
        ]
    )

    return music_transformer

def train(model, train_data, valid_data, epochs, model_save_dir='model/weights/'): 

    print("training model")

    # checkpoint_dir = os.path.dirname(model_save_dir)

    for (wav, midi_in), midi_out in train_data: 
        print(wav.shape, midi_in.shape, midi_out.shape)
        
        for i in range(wav.shape[0]): 
            with tf.GradientTape() as tape: 

                pred = model((wav[i, :, :], midi_in[i, :, :]))

                print(pred.shape)

                loss = model.loss(pred, midi_out[i, :, :])


            gradients = tape.gradient(loss, model.trainable_weights)

            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_dir,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)

    # history = model.fit(train_data,   
    #         epochs = epochs,
    #         validation_data = valid_data, 
    #         callbacks=[cp_callback]) 
    
    # print("training finished")
    
    return None
    

