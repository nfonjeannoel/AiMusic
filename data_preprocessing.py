import glob
import pickle
import numpy as np
from keras.utils import np_utils
from music21 import converter, instrument, note, chord


def get_notes():
    notes = []
    for file in glob.glob("data/*.mid"):
        midi = converter.parse(file)

        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    # Create a dictionary to map pitches to integers
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input sequences into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    # Convert the output to one-hot encoding
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)
