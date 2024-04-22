import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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
def wav_to_spectrogram(wav_file, hop_length = 4096, fmin = 65.4, n_bins = 48, window_size = 7):
    """
    Converts a wav file into a tensor input for transformer model
    param wav: path to a wav file 
    """
    y, sr = librosa.load(wav_file)
    spectrogram = librosa.cqt(y, sr = sr, hop_length = hop_length, fmin = fmin, n_bins = n_bins) 
    
    print(spectrogram.shape) #(num_freq_bins, num_time_frames)
    spectrogram = spectrogram.T #transpose spectrogram to get right order for transformer
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref = np.max) #convert to dB scale 
    minDB = np.min(spectrogram)
    
    print(f'Minimum: {np.min(spectrogram)}, Maximum: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)}') 
    spectrogram = np.pad(spectrogram, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)  
    print(spectrogram.shape)

    windows = []
    for i in range(spectrogram.shape[0] - window_size + 1):
        w = spectrogram[i:i + window_size, :]
        windows.append(w)
    
    x = np.array(windows)
    print(f'padded spectrogram and windowed shape: {x.shape}')

    #plot_spectrogram(spectrogram_db, sr)
    return spectrogram

def plot_spectrogram(spectrogram, sr):
    """
    Creates a visualization of a spectrogram, for testing purpose
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('CQT Spectrogram')
    plt.tight_layout()
    plt.show()
    
def main():
    #convert_mp3_to_wav("data/mp3", "data/wav") 
    spectrogram1 = wav_to_spectrogram("data/wav/bach1.wav")
    spectrogram2 = wav_to_spectrogram("data/wav/bach2.wav")

if __name__ == '__main__':
    main()