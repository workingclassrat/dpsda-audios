"""
Embedding function Phi(x)

    d(x, z) = || Phi(x) - Phi(z) ||_2                      (Eq. 1)

PANNs' CNN14 (CNN trained on AudioSet for audio-event tagging) is used as Phi and plays the role Inception-v3 played for images:
classifier whose penultimate-layer activations serve as a semantically meaningful embedding space for computing distances between samples.

runs each audio file through PANNs to get a vector representing location of clip in embedding space
"""

__version__ = "v3"

import librosa
import numpy as np
import torch
from panns_inference import AudioTagging

_device = "cuda" if torch.cuda.is_available() else "cpu"
# Loading with checkpoint_path=None downloads PANNs' default pretrained CNN14
# checkpoint the first time this module is imported
_model = AudioTagging(checkpoint_path=None, device=_device)

_MIN_DURATION_S = 2.0


def load_and_resample(wav_path: str, target_sr: int = 32000) -> np.ndarray:
    """
    Load a .wav file from disk, resample it to PANNs' expected 32kHz, and zero-pad it up to _MIN_DURATION_S if it's shorter than that
    
    padding with silence doesn't change what event embed() perceives (PANNs pools over time); it just guarantees the network has enough frames to survive all 5 of CNN14's internal pooling stages
    """
    waveform, _ = librosa.load(wav_path, sr=target_sr, mono=True)
    min_samples = int(target_sr * _MIN_DURATION_S)
    if len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))
    return waveform.astype(np.float32)


def embed(waveform: np.ndarray) -> np.ndarray:
    """
    waveform: 1D float32 array, mono, 32kHz (use load_and_resample() first)
    """
    batched = waveform[None, :]                    # turns 1D array into 2D array since only one clip is embedded and PANNs expects a batch of clips/2D array in shape (1, num_samples)
    _, embedding = _model.inference(batched)        # embedding shape: (1, 2048)
    return embedding[0]     # 2048-d embedding vector (Phi(x) for one sample)


def embed_paths(wav_paths: list) -> np.ndarray:
    """
    load + embed a list of .wav file paths -> (N', 2048) array. (N' is number of files that successfully load)
    """
    embeddings = []
    skipped = []
    for p in wav_paths:
        try:
            embeddings.append(embed(load_and_resample(p)))
        except Exception as e:
            skipped.append(p)
            print(f"  WARNING: skipping unreadable audio file '{p}' ({type(e).__name__}: {e})")

    if not embeddings:
        raise RuntimeError(
            f"None of the {len(wav_paths)} provided audio files could be loaded -- "
            "check that the dataset downloaded/extracted correctly."
        )
    if skipped:
        print(f"  ({len(skipped)}/{len(wav_paths)} files skipped due to load errors)")

    return np.stack(embeddings)
