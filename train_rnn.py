import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from data_preprocessing import prepare_sequences, get_notes


def train_rnn(network_input, network_output, model_path):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(network_output.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train the model
    model.fit(network_input, network_output, epochs=200, batch_size=128)

    # Save the model to disk
    model.save(model_path)


if __name__ == '__main__':
    notes = get_notes()
    network_input, network_output = prepare_sequences(notes, len(set(notes)))

    # with open('notes.pkl', 'rb') as filepath:
    #     notes = pickle.load(filepath)

    # n_vocab = len(set(notes))

    # network_input, network_output = prepare_sequences(notes, n_vocab)

    train_rnn(network_input, network_output, 'model.h5')
