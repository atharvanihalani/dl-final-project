from music_transformer import MusicDecoderTransformer

def main():
    print("testing model instantiation")

    test_model = MusicDecoderTransformer(
        units = 88,
        window_size = 128, 
        embed_size = 512,
        )
    

    test_model.build()
    
    test_model.summary()

    print(test_model)

if __name__ == "__main__": 
    main()