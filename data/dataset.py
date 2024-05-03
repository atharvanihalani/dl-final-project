import tensorflow as tf
from preprocess import wav_to_spectrogram, midi_to_piano_roll
from preprocess_constants import SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE, START_TOKENS, END_TOKENS

@tf.py_function(Tout=tf.float32)
def process_example(wav_file, midi_file): 
    input = wav_to_spectrogram(wav_file.to_string(), SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE)
    label = midi_to_piano_roll(midi_file.to_string(), SAMPLING_RATE)
    return tf.Tensor(input), tf.Tensor(label)

def load_data(dataset = 'saarland') -> tf.data.Dataset: 

    path_wav = ('data/'+ dataset + '/wav/*.wav')
    path_midi = ('data/'+ dataset + '/midi/*.mid')

    wav = tf.data.Dataset.list_files(path_wav, shuffle=False)
    midi = tf.data.Dataset.list_files(path_midi, shuffle=False)

    ds = tf.data.Dataset.zip((wav, midi))

    for x, y in ds:
        print(x, y)

    # ds = ds.map(process_example)

    # for x, y in ds:
    #     print(x.shape, y.shape)

    return ds

if __name__ == "__main__": 
    ds = load_data()