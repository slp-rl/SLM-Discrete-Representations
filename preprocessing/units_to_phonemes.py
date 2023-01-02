import glob
import os

import numpy as np
import pandas as pd
import torchaudio
import tqdm

from preprocessing.encoders import ENCODERS
from preprocessing.phonemes_manager import PHONEMES, phoneme_to_index, index_to_phoneme
from preprocessing.utils import get_model_units_seconds, SR


def get_phonemes_units(dense_model:str, vocab:int):
    """
    Construct a mapping between audio processing units and phonemes by iterating through the audio files in the TIMIT dataset
    and using a provided encoder object to convert the audio waveform of each file into units. The function then reads the
    corresponding phoneme transcription file for each audio file and uses the start and end times of each phoneme in the
    transcription to determine which units correspond to which phonemes. The function increments a count in the mapping array
    for each unit-phoneme pair.
    """
    encoder = ENCODERS[dense_model](vocab)

    unit_len = get_model_units_seconds(dense_model)
    files = glob.glob(os.path.join("datasets/TIMIT", "*", "*", "*", "*", "*wav"))
    assert files, "TIMIT dataset is empty"
    units_to_phonemes = np.zeros((vocab, len(PHONEMES.keys())))
    for wav_file in tqdm.tqdm(files):
        # Load audio waveform and convert to desired sample rate
        waveform, sample_rate = torchaudio.load(wav_file)
        if sample_rate != SR:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SR)
        # Encode audio waveform into units
        units = encoder.encode(waveform)

        # Read phoneme transcription file
        p_file = wav_file.replace("WAV.wav", "PHN")
        with open(p_file) as f:
            data = [x.split() for x in f.read().splitlines()]

        # Iterate through phoneme transcriptions and determine corresponding units
        for i, (start, end, phoneme) in enumerate(data):
            start_index = int((float(start) / SR) / unit_len)
            end_index = int(((float(end) / SR) / unit_len) - 0.5) + 1
            start_index += 1
            end_index -= 1
            for unit in units[start_index:end_index + 1]:
                units_to_phonemes[unit, phoneme_to_index(phoneme)] += 1
    return units_to_phonemes


def save_units_phonemes(dense_model='hubert', vocab=100, n=3):
    """
    Saves the unit-to-phoneme mapping and the top n phonemes for each unit in two separate CSV files.
    """
    # retrieve the unit-to-phoneme mapping
    units_to_phonemes = get_phonemes_units(dense_model, vocab)

    # create the assets/units_phonemes directory if it doesn't exist
    os.makedirs("assets/units_phonemes/", exist_ok=True)

    # save the unit-to-phoneme mapping to a CSV file
    pd.DataFrame(units_to_phonemes).to_csv(f"assets/units_phonemes/{dense_model}_{vocab}_counts.csv")

    # create a data frame to store the top n phonemes for each unit
    columns = []
    for i in range(n):
        columns.extend([f'top_{i}_index', f'top_{i}_phoneme', f'top_{i}_percentage'])
    stats = pd.DataFrame(index=range(vocab), columns=columns)

    # populate the data frame with the top n phonemes for each unit
    for u in range(vocab):
        unit_counts = units_to_phonemes[u, :]
        top_n_indexes = np.argsort(unit_counts)[::-1][:n]
        top_n_phonemes = [index_to_phoneme(i) for i in top_n_indexes]
        top_n_percentage = unit_counts[top_n_indexes] / unit_counts.sum()
        stats.loc[u, [f'top_{i}_index' for i in range(n)]] = top_n_indexes
        stats.loc[u, [f'top_{i}_phoneme' for i in range(n)]] = top_n_phonemes
        stats.loc[u, [f'top_{i}_percentage' for i in range(n)]] = top_n_percentage
    os.makedirs("assets/units_phonemes", exist_ok=True)
    stats.to_csv(f"assets/units_phonemes/{dense_model}_{vocab}_stats.csv")
