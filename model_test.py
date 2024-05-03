from model.music_transformer import MusicDecoderTransformer
from data.preprocess import wav_to_spectrogram, midi_to_piano_roll
from data.globals import SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE, START_TOKENS, END_TOKENS

def main():
    print("testing model instantiation")

    cqt1 = wav_to_spectrogram("data/saarland/wav/Bach_BWV849-01_001_20090916-SMD.wav", SAMPLING_RATE, SECONDS, BINS_PER_OCTAVE)
    print(cqt1.shape)
    pm1 = midi_to_piano_roll("data/saarland/midi/Bach_BWV849-01_001_20090916-SMD.mid", SAMPLING_RATE)

    test_model = MusicDecoderTransformer(
        units = 88,
        window_size = 217, 
        embed_size = 512,
    )

    test_model.build(input_shape=(32, 217 , 84))

    test_model.summary()

    out = test_model(cqt1)
    
    print(out.shape)

if __name__ == "__main__": 
    main()