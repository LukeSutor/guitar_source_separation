from asyncio.windows_events import NULL
import os
import math
import random
import torchaudio
import torch

DEVICE = "cpu"
SAMPLE_RATE = 44100
GUITAR_DIR = 'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/guitar/'
INTERFERERS_DIR = 'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/interferers/'
ROOT_DIR = 'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/'

def transform_and_save(filename, label, device=DEVICE):
    audio_dir = (GUITAR_DIR if label == 'guitar' else INTERFERERS_DIR) + filename
    signal, sr = torchaudio.load(audio_dir)

    signal = signal.to(device)
    signal = _mix_down_if_necessary(signal)
    signal = _resample_if_necessary(signal, sr, device)

    train_sample = random.random() >= 0.2

    save_dir = "train/{}/{}".format(label, filename) if train_sample else "valid/{}/{}".format(label, filename)

    torchaudio.save(
        ROOT_DIR+save_dir,
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

def total_length_seconds(directory):
    time = 0

    for file in os.listdir(directory):
        if file.endswith('.wav'):
            signal, sr = torchaudio.load(directory+file)
            time += signal.shape[1] / sr

    return math.floor(time * 100) / 100

def split_dataset():
    # Make directories
    directories = [
        'train',
        'valid',
        'train/guitar',
        'train/interferers',
        'valid/guitar',
        'valid/interferers'
    ]

    for dir in directories:
        try:
            os.mkdir(ROOT_DIR+dir)
        except Exception as e:
            print(e)

    # Go through files in guitar samples and transform/save them, 80/20 train/valid split
    for file in os.listdir(GUITAR_DIR):
        if file.endswith('.wav'):
            transform_and_save(file, 'guitar')

    # Go through files in interferer samples and transform/save them, 80/20 train/valid split
    for file in os.listdir(INTERFERERS_DIR):
        if file.endswith('.wav'):
            transform_and_save(file, 'interferers')

    print("fin")


if __name__ == "__main__":
    split_dataset()