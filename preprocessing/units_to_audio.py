import dataclasses
import os
from typing import Dict

import numpy as np
import torchaudio
import tqdm
from soundfile import write

from preprocessing.encoders import ENCODERS
from preprocessing.utils import get_model_units_seconds, SR


@dataclasses.dataclass
class UnitItem:
    """
    A class representing a unit in a dataset.

    Attributes:
    - unit: int - the unit number (default: -1)
    - file_index: int - the index of the file the unit belongs to (default: 0)
    - length: int - the length of the unit (default: 0)
    - index_start: int - the start index of the unit (default: 0)
    - audio: np.ndarray - the audio data for the unit (default: np.array([]))
    """
    unit: int = -1
    file_index: int = 0
    length: int = 0
    index_start: int = 0
    audio: np.ndarray = np.array([])


def index_to_data(data, index, unit_count, length: int = 1, max_len=10):
    """
    Given a wav form data, an index, and a unit count, returns a slice of the data
    starting from the given index and going up to the given length.
    The returned slice is limited by the max_len parameter if the requested
    length is greater than max_len.
    """
    length = min(max_len, length)
    data = data.flatten()
    return data[index * unit_count:(index + length) * unit_count].cpu().numpy().flatten()


from typing import List


def update_items_list(list_: List[UnitItem], item: UnitItem, max_len=100):
    """
    Updates a list of `UnitItem`s by adding or replacing an item.

    If the list is shorter than the maximum length, the item is added to the list.
    Otherwise, the item with the minimum length is replaced if the new item has a
    greater length.

    """
    if len(list_) < max_len:
        list_.append(item)
    else:
        elements_length = [x.length for x in list_]
        min_index = elements_length.index(min(elements_length))
        if item.length > elements_length[min_index]:
            list_[min_index] = item


def fill_item_audio(dataset: torchaudio.datasets, item: UnitItem, unit_count):
    """
    Fills the `audio` attribute of a `UnitItem` with data from a dataset.

    The function retrieves the waveform and sample rate of the file corresponding
    to the `file_index` attribute of the `UnitItem`, and resamples the waveform to
    the target sample rate if necessary. It then slices the waveform using the
    `index_start` and `length` attributes of the `UnitItem` and stores the result
    in the `audio` attribute.
    """
    waveform, sample_rate, *_ = dataset[item.file_index]
    if sample_rate != SR:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SR)
    item.audio = index_to_data(waveform, item.index_start, unit_count, item.length)


def save_units_audio(dense_model='hubert', vocab=100, ds="LJSPEECH"):
    """
    Saves the audio data for all units in a vocabulary to a file.

    This function loads a dataset and encodes the audio data in the dataset using
    the specified encoder. It then processes the encoded data to create a list of
    `UnitItem`s for each unit in the vocabulary. Finally, it fills the `audio`
    attribute of each `UnitItem` with data from the dataset, concatenates the
    `audio` data for each unit, and saves it to a file.
    """
    encoder = ENCODERS[dense_model](vocab)
    unit_count = int(get_model_units_seconds(dense_model) * SR)

    units_map: Dict[int, List[UnitItem]] = {i: [] for i in range(vocab)}
    root = f"datasets/{ds}"
    os.makedirs(root, exist_ok=True)

    if ds == 'LIBRISPEECH':
        dataset = torchaudio.datasets.LIBRISPEECH(root=root, download=True)
    else:  # LJSPEECH
        dataset = torchaudio.datasets.LJSPEECH(root=root, download=True)

    for file_index, (waveform, sample_rate, *_) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if sample_rate != SR:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SR)
        units = encoder.encode(waveform)
        unit_item = UnitItem(file_index=file_index, unit=units[0], index_start=0)
        for i_, unit in enumerate(units[1:]):
            index = i_ + 1
            if unit == unit_item.unit:
                unit_item.length += 1
            else:
                update_items_list(units_map[unit_item.unit], unit_item)
                unit_item = UnitItem(file_index=file_index, unit=unit, index_start=index)
    output_base = f"assets/audio/{ds}_{vocab}_{dense_model}"
    os.makedirs(output_base, exist_ok=True)
    for unit, item_list in units_map.items():
        for item in item_list:
            fill_item_audio(dataset, item, unit_count)
        if len(units_map[unit]):
            audio = np.concatenate([list(x.audio) for x in units_map[unit]])
            write(os.path.join(output_base, f"{unit}.wav"), audio, SR)
            units_map[unit] = [UnitItem()]
