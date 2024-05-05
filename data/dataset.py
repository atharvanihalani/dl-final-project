import tensorflow as tf
from data.preprocess import wav_to_spectrogram, midi_to_piano_roll
from data.preprocess_constants import SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE

def process_example(wav_file, midi_file) -> tf.data.Dataset: 
    """
    Function applied to each filename to generate wav-pianoroll pairs
    """
    # read file and generate spectrograms and piano rolls
    wav = wav_to_spectrogram(tf.compat.path_to_str(wav_file), SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE)
    midi = midi_to_piano_roll(midi_file, SAMPLING_RATE)

    # to deal with sizing effects due to spectrogram algorithm
    if wav.shape[0] != midi.shape[0]: 
        batch_dim = min(wav.shape[0], midi.shape[0])
        wav = wav[:batch_dim, :, :]
        midi = midi[:batch_dim, :, :]

    # create datasets to remove the first dimension of the tensor
    wav_data = tf.data.Dataset.from_tensor_slices(wav)
    midi_data = tf.data.Dataset.from_tensor_slices(midi)

    # and zip to organise like tuples (wav, pianoroll)
    return tf.data.Dataset.zip((wav_data, midi_data))

def create_data(dataset = 'saarland') -> tf.data.Dataset: 
    """
    Instantiates a dataset object from the input directory
    """

    path_wav = ('data/'+ dataset + '/wav/*.wav')
    path_midi = ('data/'+ dataset + '/midi/*.mid')

    wav = tf.data.Dataset.list_files(path_wav, shuffle=False)
    midi = tf.data.Dataset.list_files(path_midi, shuffle=False)

    processed = tf.data.Dataset.zip((wav, midi))

    processed = processed.flat_map(process_example)

    processed.save('data/saarland/saved/')

    # train_data, test_data = tf.keras.utils.split_dataset(processed, left_size = 0.9, shuffle=True)

    # train_data, valid_data = tf.keras.utils.split_dataset(train_valid_data, left_size = 0.8, shuffle=True)

    # train_data = train_data.shuffle(50, reshuffle_each_iteration=True).batch(32)
    # valid_data = valid_data.shuffle(50, reshuffle_each_iteration=True).batch(20)
    # test_data = test_data.shuffle(50, reshuffle_each_iteration=True).batch(32)

    return 

def prepare_batch(wav, midi): 

    # context = wav
    
    # inputs = midi[:-1, :] # remove the last token
    # labels = midi[1:, :] # remove the first token

    context = tf.reshape(wav[1:-1, :], shape=(5,43,84))
    midi = tf.reshape(midi, shape=(5, 22050, 88))

    inputs = midi[:, :-1, :]
    labels = midi[:, 1:, :]


    return (context, inputs), labels
    

def load_data(dataset = 'saarland') -> tf.data.Dataset: 

    path = "data/" + dataset + "/saved/"

    num_samples = 3414

    wav_tensor_spec = tf.TensorSpec(shape=(217, 84), dtype=tf.float32)
    pianoroll_tensor_spec = tf.TensorSpec(shape=(110250, 88), dtype=tf.float32)
    element_spec = (wav_tensor_spec, pianoroll_tensor_spec)

    data = tf.data.Dataset.load(path, element_spec=element_spec).shuffle(buffer_size=50)
    
    train_data = data.take(int(0.8 * num_samples))
    valid_test_data = data.skip(int(0.8 * num_samples))

    print(train_data, valid_test_data)

    valid_data = valid_test_data.take(int(0.1 * num_samples))
    test_data = valid_test_data.skip(int(0.1 * num_samples))

    # print(valid_data, test_data)

    train_data = train_data.shuffle(10, reshuffle_each_iteration=True)\
        .map(prepare_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    valid_data = valid_data.shuffle(20, reshuffle_each_iteration=True).batch(10).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.shuffle(20, reshuffle_each_iteration=True).batch(10).prefetch(tf.data.AUTOTUNE)

    # print(train_data, valid_data, test_data)

    return train_data, valid_data, test_data


