from asyncio.windows_events import NULL
import os
import torchaudio
import torch

DEVICE = "cpu"
SAMPLE_RATE = 44100

def Transform(audio_dir, save_dir, device=DEVICE):
    signal, sr = torchaudio.load(audio_dir)

    signal = signal.to(device)
    signal = _mix_down_if_necessary(signal)
    signal = _resample_if_necessary(signal, sr, device)

    torchaudio.save(
        save_dir,
        signal,
        SAMPLE_RATE
    )


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def _resample_if_necessary(signal, sr, device):
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sr, SAMPLE_RATE).to(device)
        signal = resampler(signal)
    return signal


if __name__ == "__main__":
    for file in os.listdir('C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/newdata'):
        if file.endswith('.wav'):
            Transform(
                'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/newdata/'+file,
                'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/newdata_cleaned/'+file
            )