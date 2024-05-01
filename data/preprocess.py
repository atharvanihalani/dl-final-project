import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from globals import SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE, START_TOKENS, END_TOKENS

def lame_process(input_file, output_file):
    '''
    Converts a single MP# file to WAV format using LAME decoder
    param input_file: path to an input MP3 file
    param output_file: path to save the output WAV file
    '''
    os.system(f'lame --decode --quiet "{input_file}" "{output_file}"')

def convert_mp3_to_wav(input_path, output_path):
    """
    Convert all MP3 files in a directory or a single MP3 file to WAV format
    param input_path: Path to the directory containing MP3 files
    param output_path: Path to save the converted WAV files to
    """
    if os.path.isdir(input_path):
        input_files = [file for file in os.listdir(input_path) if file.lower().endswith('.mp3')]
        for file_name in input_files:
            print(f'Processing {input_path}/{file_name}')
            output_file = os.path.splitext(file_name)[0] + '.wav'
            lame_process(os.path.join(input_path, file_name), os.path.join(output_path, output_file))
    else:
        lame_process(input_path, output_path)

def wav_to_spectrogram(wav_file, sr, seconds, bins_per_octave):
    """
    Converts a wav file into a tensor input for transformer model
    param wav: path to a wav file 
    """
    y, s = librosa.load(wav_file)
    print(f'sampling rate: {sr}')

    spectrogram = librosa.cqt(y, sr = sr, bins_per_octave = bins_per_octave) #only have to define either n_bins or bins_per_octave
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref = np.max) #convert to dB scale 

    #think about the tokens for <start> <token> 
    
    #print(f'spec shape: {spectrogram.shape}') #(num_freq_bins, num_time_frames)
    spectrogram = spectrogram.T #transpose spectrogram to get right order for transformer (num_time_frames, num_freq_bins)
    #print(f'spec shape: {spectrogram.shape}')
    #plot_spectrogram(spectrogram, sr)
    
    minDB = np.min(spectrogram)
    
    #print(f'Minimum: {np.min(spectrogram)}, Maximum: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)}') 
    #spectrogram = np.pad(spectrogram, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)  #pad audio
    
    window_size = librosa.time_to_frames(seconds, sr = sr)
    windows = []

    pad_width = ((0, window_size - (spectrogram.shape[0] % window_size)), (0,0))
    spectrogram = np.pad(spectrogram, pad_width, 'constant', constant_values=minDB)
    #print(f'Padded spectrogram shape: {spectrogram.shape}')
    
    for i in range(0, spectrogram.shape[0], window_size):
        w = spectrogram[i:i + window_size, :]
        windows.append(w)
    
    windows = np.array(windows) 

    #Add start and end tokens to the beginning and end of each sequence
    pad_start_tokens = ((0,0), (1, 0), (0, 0))
    windows = np.pad(windows, pad_start_tokens, 'constant', constant_values= START_TOKENS)

    pad_end_tokens = ((0,0), (0, 1), (0, 0))
    windows = np.pad(windows, pad_end_tokens, 'constant', constant_values= END_TOKENS)

    #plot_spectrogram(spectrogram_db, sr) '''
    return windows

def plot_spectrogram(spectrogram, sr):
    """
    Creates a visualization of a spectrogram, for testing purpose
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='freq')
    plt.colorbar(format='%+2.0f dB')
    plt.title('CQT Spectrogram')
    plt.tight_layout()
    plt.show()

def midi_to_piano_roll(midi_file_path, sr, start_pitch=19, end_pitch=107):
    '''
    Returns an np array of the piano roll representation of a midi file, 
    with 88 notes representing those of a piano keyboard rather than 
    the default 128 notes, that is, from MIDI note 21 (A0) to MIDI note 108 (C8).
    '''
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    raw_piano_roll = midi_data.get_piano_roll(fs=sr)[start_pitch:end_pitch]
    # 0 meaning not played -> converting it into binary representation
    piano_roll = raw_piano_roll > 0
    piano_roll = np.asarray(piano_roll).astype(int).T
    #print(f'shape of unpadded pianoroll: {piano_roll.shape}')
    #print(f'size of unpadded pianoroll in GB: {piano_roll.itemsize * piano_roll.size}')


    remainder = SECONDS*SAMPLING_RATE - (piano_roll.shape[0] % (SECONDS*SAMPLING_RATE))
    pad_width = ((0, remainder), (0,0))
    piano_roll = np.pad(piano_roll, pad_width, 'constant', constant_values=0)
    piano_roll = np.reshape(piano_roll, (-1, SECONDS * SAMPLING_RATE, piano_roll.shape[1]))
    #print(f'shape of pianoroll after splitting: {piano_roll.shape}')
    #print(f'size of padded/split pianoroll in GB: {piano_roll.itemsize * piano_roll.size }')
    return piano_roll

    
def main():
    # convert_mp3_to_wav("data/saarland/mp3", "data/saarland/wav") 
    cqt1 = wav_to_spectrogram("data/saarland/wav/Bach_BWV849-01_001_20090916-SMD.wav", SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE)
    print(cqt1.shape)
    pm1 = midi_to_piano_roll("data/saarland/midi/Bach_BWV849-01_001_20090916-SMD.mid", SAMPLING_RATE)
    print(pm1.shape)

if __name__ == '__main__':
    main()