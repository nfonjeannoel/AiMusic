import numpy as np
import pickle
import tensorflow as tf
from music21 import converter, instrument, note, stream, chord
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model.h5')

# Load the notes used during training
with open('notes.pkl', 'rb') as filepath:
    notes = pickle.load(filepath)

n_vocab = len(set(notes))

# Create a dictionary to map pitches to integers
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# Set the sequence length and seed notes
sequence_length = 100

import random


def generate_seed_notes(notes, sequence_length):
    seed_notes = []
    for i in range(sequence_length):
        seed_note = np.random.choice(notes)
        seed_notes.append(seed_note)
    return seed_notes

seed_notes = generate_seed_notes(notes, sequence_length)

# Generate music
generated_notes = []
for i in range(500):
    # Create the input sequence
    input_sequence = [note_to_int[note] for note in seed_notes]
    input_sequence = np.reshape(input_sequence, (1, sequence_length, 1))
    input_sequence = input_sequence / float(len(set(notes)))

    # Make a prediction
    prediction = model.predict(input_sequence, verbose=0)

    # Get the index of the predicted note
    index = np.argmax(prediction)

    # Convert the index back to the corresponding note
    result = pitchnames[index]

    # Add the predicted note to the generated notes list
    generated_notes.append(result)

    # Update the seed notes
    seed_notes = seed_notes[1:] + [result]

# Create a stream object for the generated notes and write it to disk
offset = 0
output_notes = []
for pattern in generated_notes:
    # Handle chords
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(current_note)
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # Handle notes
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # Increase the offset
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output.mid')
