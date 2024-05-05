import tensorflow as tf
from model.model import get_compiled_model, train
from data.dataset import load_data
import pickle


def main():
    model = get_compiled_model()
    # model.build(input_shape=((32, 217, 84), (32, 110250, 88)))
    # model.summary()

    train_data, valid_data, test_data = load_data()
    print("data loading finished")

    history = train(model, train_data, valid_data, 1)
    print(history)

    with open('model/train-history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__": 
    main()
