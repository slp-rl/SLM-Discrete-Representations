from abc import ABC, abstractmethod

import joblib
import numpy as np
import torch
from textless.data.speech_encoder import SpeechEncoder
from torchaudio.compliance import kaldi as kaldi


class Encoder(ABC):
    """
    Abstract class for defining an encoder.
    """

    @abstractmethod
    def __init__(self, vocab: int) -> None:
        """
        Initializes the encoder with a vocabulary size.
        """
        pass

    @abstractmethod
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """
        Encodes a waveform and returns the resulting units as a NumPy array.
        """
        pass


class HuBERTEncoder(Encoder):
    def __init__(self, vocab: int) -> None:
        """
        Initializes the HuBERT encoder with a vocabulary size and loads the model onto the device (GPU if available, otherwise CPU).
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = SpeechEncoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab,
            deduplicate=False,
            need_f0=False,
            f0_normalizer=None,
            f0_quantizer=None,
        ).to(self.device)

    def encode(self, waveform):
        """
        Encodes a waveform using the HuBERT model and returns the resulting units as a NumPy array.
        """
        encoded = self.encoder(waveform.to(self.device))
        units = encoded["units"]
        return units.cpu().numpy()


class CPCEncoder(Encoder):
    def __init__(self, vocab):
        """
        Initializes the CPC encoder with a vocabulary size and loads the model onto the device (GPU if available, otherwise CPU).
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = SpeechEncoder.by_name(
            dense_model_name='cpc-big-ll6k',
            quantizer_model_name='kmeans',
            vocab_size=vocab,
            deduplicate=False,
            need_f0=False,
            f0_normalizer=None,
            f0_quantizer=None,
        ).to(self.device)

    def encode(self, waveform):
        """
        Encodes a waveform using the CPC model and returns the resulting units as a NumPy array.
        """
        encoded = self.encoder(waveform.to(self.device))
        units = encoded["units"]
        return units.cpu().numpy()

class LogMelEncoder(Encoder):

    def __init__(self, vocab, num_mel_bins=80, frame_length=25.0):
        """
        Initializes the LogMel encoder with a vocabulary size and loads the model onto the device (GPU if available, otherwise CPU).
        """
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        kmeans_file = f"assets/kmeans/km_logmel_{vocab}.bin"
        self.km = joblib.load(kmeans_file)

    def encode(self, waveform):
        """
        Encodes a waveform using the LogMel model and returns the resulting units as a NumPy array.
        """
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(waveform).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            sample_frequency=16_000,
        )
        units = np.argmin(self.km.transform(feat.cpu()), axis=1)
        return units


ENCODERS = {
    'hubert': HuBERTEncoder,
    'cpc': CPCEncoder,
    'logmel': LogMelEncoder,
}
