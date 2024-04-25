import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

SAMPLING_RATE = 22050

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

#base minimum frequency off of midi min freq? 
#need to iron out the inputs for the cqt 
def wav_to_spectrogram(wav_file, sr = SAMPLING_RATE, spec_type = "cqt", hop_length = 512):
    """
    Converts a wav file into a tensor input for transformer model
    param wav: path to a wav file 
    """
    y, s = librosa.load(wav_file)
    print(f'sampling rate: {sr}')

    if spec_type == "cqt":
        spectrogram = librosa.cqt(y, sr = sr, bins_per_octave = 36) #only have to define either n_bins or bins_per_octave
    else:
        spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 512)

    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref = np.max) #convert to dB scale 

    #think about the tokens for <start> <token> 
    
    print(f'spec shape: {spectrogram.shape}') #(num_freq_bins, num_time_frames)
    spectrogram = spectrogram.T #transpose spectrogram to get right order for transformer (num_time_frames, num_freq_bins)
    print(f'spec shape: {spectrogram.shape}')
    #plot_spectrogram(spectrogram, sr)
    
    minDB = np.min(spectrogram)
    
    #print(f'Minimum: {np.min(spectrogram)}, Maximum: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)}') 
    #spectrogram = np.pad(spectrogram, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)  #pad audio
    
    window_size = librosa.time_to_frames(5, sr = sr)
    windows = []

    pad_width = ((0, window_size - (spectrogram.shape[0] % window_size)), (0,0))
    spectrogram = np.pad(spectrogram, pad_width, 'constant', constant_values=minDB)
    print(f'Padded spectrogram shape: {spectrogram.shape}')

    i = 0
    start_frames = []

    for i in range(0, spectrogram.shape[0], window_size):
        w = spectrogram[i:i + window_size, :]
        start_frames.append(i)
        windows.append(w)
    
    windows = np.array(windows) 
    start_frames = np.array(start_frames)
    print(f'windowed shape: {windows.shape}')

    #plot_spectrogram(spectrogram_db, sr) '''
    return windows, start_frames

def get_times(frames, sr):
    times = librosa.frames_to_time(frames, sr=sr)
    return times 

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

def midi_to_np(midi_file):
    pm_mid = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = pm_mid.get_piano_roll(fs=SAMPLING_RATE).T
    piano_roll[piano_roll > 0] = 1
    print(f'pianoroll non-padded shape: {piano_roll.shape[0]}')

    remainder = 5*SAMPLING_RATE - (piano_roll.shape[0] % (5*SAMPLING_RATE))
    pad_width = ((0, remainder), (0,0))
    print(pad_width)
    piano_roll = np.pad(piano_roll, pad_width, 'constant', constant_values=0)
    print(f'pianoroll padded shape: {piano_roll.shape}')
    piano_roll = np.reshape(piano_roll, (-1, 5 * SAMPLING_RATE, piano_roll.shape[1]))
    return piano_roll

    
def main():
    #convert_mp3_to_wav("data/mp3", "data/wav") 
    cqt1, start_frames = wav_to_spectrogram("data/wav/bach1.wav", spec_type= "mel")
    print(get_times(start_frames, SAMPLING_RATE)[:5])
    start_times = librosa.frames_to_time(start_frames, sr=SAMPLING_RATE)
    
    #print(f'times max: {np.max(times)}, times:min {np.min(times)}')
    #print(f'times shape: {times.shape}')

    pm = pretty_midi.PrettyMIDI("data/midi/bach1.mid")
    #times = range(start_times[0], start_times[1], 1/SAMPLING_RATE)
    #print(len(times))

    pianoroll = pm.get_piano_roll(fs=SAMPLING_RATE,times=None) 
    pianoroll_time = midi_to_np("data/midi/bach1.mid")

    print(f'pianoroll w/o time: {pianoroll.shape}')
    print(f'pianoroll w/ time: {pianoroll_time.shape}')

    #spectrogram2 = wav_to_spectrogram("data/wav/bach2.wav", spec_type= "mel")

if __name__ == '__main__':
    main()