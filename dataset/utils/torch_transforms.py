import os
import torchaudio
import torch

def Transform(audio_dir, save_dir):
    signal, sr = torchaudio.load(audio_dir)

    signal = _mix_down_if_necessary(signal)

    torchaudio.save(
        save_dir,
        signal,
        sr
    )


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


if __name__ == "__main__":
    for file in os.listdir('C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data'):
        if file.endswith('.wav'):
            Transform(
                'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/'+file,
                'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/guitar/'+file
            )