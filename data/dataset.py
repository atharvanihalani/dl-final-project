import tensorflow as tf
from preprocess import wav_to_spectrogram, midi_to_piano_roll
from preprocess_constants import SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE

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

def load_data(dataset = 'saarland') -> tf.data.Dataset: 
    """
    Instantiates a dataset object from the input directory
    """

    path_wav = ('data/'+ dataset + '/wav/*.wav')
    path_midi = ('data/'+ dataset + '/midi/*.mid')

    wav = tf.data.Dataset.list_files(path_wav, shuffle=False)
    midi = tf.data.Dataset.list_files(path_midi, shuffle=False)

    # wav = wav.flat_map(tf.data.Dataset.from_tensors)
    # midi = midi.flat_map(tf.data.Dataset.from_tensors)

    processed = tf.data.Dataset.zip((wav, midi))

    processed = processed.flat_map(process_example)

    processed = processed.shuffle(50, reshuffle_each_iteration=True).batch(20)

    return processed