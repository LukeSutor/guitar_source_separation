from typing import Optional, Union

import torch
import os
import sys
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json

print(sys.path[0])

from model import OpenUnmix as openunmix
import model

import random
from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain, Phaser, Delay, Distortion, PitchShift


def bandwidth_to_max_bin(rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns:
        np.ndarray: maximum frequency bin
    """
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(state: dict, is_best: bool, path: str, target: str):
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        state (dict): torch model state dict
        is_best (bool): if current model is about to be saved as best model
        path (str): model path
        target (str): target name
    """
    # save full checkpoint including optimizer
    torch.save(state, os.path.join(path, target + ".chkpnt"))
    if is_best:
        # save just the weights
        torch.save(state["state_dict"], os.path.join(path, target + ".pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


def load_target_models(targets, model_str_or_path="umxhq", device="cpu", pretrained=True, source = "pth"):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    if isinstance(targets, str):
        targets = [targets]

    model_path = Path(model_str_or_path).expanduser()
    if not model_path.exists():
        # model path does not exist, use pretrained models
        try:
            # disable progress bar
            hub_loader = getattr(openunmix, model_str_or_path + "_spec")
            err = io.StringIO()
            with redirect_stderr(err):
                return hub_loader(targets=targets, device=device, pretrained=pretrained)
            print(err.getvalue())
        except AttributeError:
            raise NameError("Model does not exist on torchhub")
            # assume model is a path to a local model_str_or_path directory
    else:
        models = {}
        for target in targets:
            # load model from disk
            with open(Path(model_path, target + ".json"), "r") as stream:
                results = json.load(stream)

            if source == "chkpnt":
                target_model_path = next(Path(model_path).glob("%s*.chkpnt" % target))
                state = torch.load(target_model_path, map_location=device)["state_dict"]
                print("loading from checkpoint")
            else:
                target_model_path = next(Path(model_path).glob("%s*.pth" % target))
                state = torch.load(target_model_path, map_location=device)


            models[target] = model.OpenUnmix(
                nb_bins=results["args"]["nfft"] // 2 + 1,
                nb_channels=results["args"]["nb_channels"],
                hidden_size=results["args"]["hidden_size"],
                max_bin=state["input_mean"].shape[0],
            )

            if pretrained:
                models[target].load_state_dict(state, strict=False)

            models[target].to(device)
        return models


def load_separator(
    model_str_or_path: str = "umxhq",
    targets: Optional[list] = None,
    niter: int = 1,
    residual: bool = False,
    wiener_win_len: Optional[int] = 300,
    device: Union[str, torch.device] = "cpu",
    pretrained: bool = True,
    filterbank: str = "torch",
    source="pth"
):
    """Separator loader

    Args:
        model_str_or_path (str): Model name or path to model _parent_ directory
            E.g. The following files are assumed to present when
            loading `model_str_or_path='mymodel', targets=['vocals']`
            'mymodel/separator.json', mymodel/vocals.pth', 'mymodel/vocals.json'.
            Defaults to `umxhq`.
        targets (list of str or None): list of target names. When loading a
            pre-trained model, all `targets` can be None as all targets
            will be loaded
        niter (int): Number of EM steps for refining initial estimates
            in a post-processing stage. `--niter 0` skips this step altogether
            (and thus makes separation significantly faster) More iterations
            can get better interference reduction at the price of artifacts.
            Defaults to `1`.
        residual (bool): Computes a residual target, for custom separation
            scenarios when not all targets are available (at the expense
            of slightly less performance). E.g vocal/accompaniment
            Defaults to `False`.
        wiener_win_len (int): The size of the excerpts (number of frames) on
            which to apply filtering independently. This means assuming
            time varying stereo models and localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
            Defaults to `300`
        device (str): torch device, defaults to `cpu`
        pretrained (bool): determines if loading pre-trained weights
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    model_path = Path(model_str_or_path).expanduser()

    # when path exists, we assume its a custom model saved locally
    if model_path.exists():
        if targets is None:
            raise UserWarning("For custom models, please specify the targets")

        target_models = load_target_models(
            targets=targets, model_str_or_path=model_path, pretrained=pretrained, source=source
        )

        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        separator = model.Separator(
            target_models=target_models,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            sample_rate=enc_conf["sample_rate"],
            n_fft=enc_conf["nfft"],
            n_hop=enc_conf["nhop"],
            nb_channels=enc_conf["nb_channels"],
            filterbank=filterbank,
        ).to(device)

    # otherwise we load the separator from torchhub
    else:
        hub_loader = getattr(openunmix, model_str_or_path)
        separator = hub_loader(
            targets=targets,
            device=device,
            pretrained=True,
            niter=niter,
            residual=residual,
            filterbank=filterbank,
        )

    return separator


def preprocess(
    audio: torch.Tensor,
    rate: Optional[float] = None,
    model_rate: Optional[float] = None,
) -> torch.Tensor:
    """
    From an input tensor, convert it to a tensor of shape
    shape=(nb_samples, nb_channels, nb_timesteps). This includes:
    -  if input is 1D, adding the samples and channels dimensions.
    -  if input is 2D
        o and the smallest dimension is 1 or 2, adding the samples one.
        o and all dimensions are > 2, assuming the smallest is the samples
          one, and adding the channel one
    - at the end, if the number of channels is greater than the number
      of time steps, swap those two.
    - resampling to target rate if necessary

    Args:
        audio (Tensor): input waveform
        rate (float): sample rate for the audio
        model_rate (float): sample rate for the model

    Returns:
        Tensor: [shape=(nb_samples, nb_channels=2, nb_timesteps)]
    """
    shape = torch.as_tensor(audio.shape, device=audio.device)

    if len(shape) == 1:
        # assuming only time dimension is provided.
        audio = audio[None, None, ...]
    elif len(shape) == 2:
        if shape.min() <= 2:
            # assuming sample dimension is missing
            audio = audio[None, ...]
        else:
            # assuming channel dimension is missing
            audio = audio[:, None, ...]
    if audio.shape[1] > audio.shape[2]:
        # swapping channel and time
        audio = audio.transpose(1, 2)
    if audio.shape[1] > 2:
        warnings.warn("Channel count > 2!. Only the first two channels " "will be processed!")
        audio = audio[..., :2]

    if audio.shape[1] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=1)

    if rate != model_rate:
        warnings.warn("resample to model sample rate")
        # we have to resample to model samplerate if needed
        # this makes sure we resample input only once
        resampler = torchaudio.transforms.Resample(
            orig_freq=rate, new_freq=model_rate, resampling_method="sinc_interpolation"
        ).to(audio.device)
        audio = resampler(audio)
    return audio



# Pedalboard audio utilities used for augmenting data 
# in the dataloader method load_audio()
# Applies random pedalboard effects to the guitar audio samples
# and less extreme effects to the interference audio samples

def create_pedalboard(file_ending, ema):
    """
    Function to create a new pedalboard object from the defined pedals below.
    Takes in the argument 'file_ending', which is the last 3 digits of the filename
    from the path passed to the dataloader. If the last 3 digits are "cln", it means
    the audio sample is from a clean guitar recording, and more extreme distortions 
    and gains are put on it.
    """


    # Pedals
    chorus_choices = [
    Chorus(centre_delay_ms = (random.choice([7,8])), depth = (0.10 + random.random() * 0.25 * ema), feedback = (0.10 + random.random() * 0.25 * ema)), # Classic chorus
    # Chorus(centre_delay_ms = (random.choice([1,2])), depth = (random.random() * 0.15 * ema), feedback=(0.7 + random.random() * 0.25 * ema)), # Flanger
    # Chorus(centre_delay_ms = (random.choice([1,2])), depth=(random.random() * 0.15 * ema), feedback=(0.7 + random.random() * 0.25 * ema), mix=1) # Vibrato
    ]

    compressor_choices = [
        Compressor()
    ]

    delay_choices = [
        Delay(delay_seconds = (random.random() * 0.5 * ema)),
        Delay(delay_seconds = (random.random() * 0.5 * ema), feedback=(0.1 + random.random() * 0.15 * ema)),
        Delay(delay_seconds = (random.random() * 0.5 * ema), mix = (0.35 + random.random() * 0.4 * ema)),
        Delay(delay_seconds = (random.random() * 0.5 * ema), feedback=(0.1 + random.random() * 0.15 * ema), mix = (0.35 + random.random() * 0.4 * ema)),
    ]

    distortion_choices = [
        Distortion(drive_db = (25 + random.random() * 35 * ema)), # High
        Distortion(drive_db = (random.random() * 25 * ema)) # Low
    ]

    gain_choices = [
        Gain(gain_db = (15 + random.random() * 15 * ema)), # High
        Gain(gain_db = (random.random() * 15 * ema)) # Low
    ]

    phaser_choices = [
        Phaser(rate_hz = (random.random() * ema), depth = (0.8 + random.random() * 0.8 * ema)),
        Phaser(rate_hz = (1 + random.random() * 2 * ema), depth = (random.random() * 0.8 * ema)),
        Phaser(rate_hz = (random.random() * ema), depth = (0.8 + random.random() * 0.8 * ema), feedback = (0.2 + random.random() * 0.5 * ema)),
        Phaser(rate_hz = (1 + random.random() * 2 * ema), depth = (random.random() * 0.8 * ema), feedback = (0.2 + random.random() * 0.5 * ema)),
    ]

    pitchshift_choices = [
        PitchShift(semitones =(random.random() * 5 * ema)), #Higher pitch
        PitchShift(semitones =(-random.random() * 5 * ema)) # Lower pitch
    ]
    
    reverb_choices = [
        Reverb(room_size = (random.random() * ema), width = (random.random() * ema), damping = (random.random() * ema)),
        Reverb(room_size = (random.random() * ema), width = (random.random() * ema), damping = (random.random() * ema), wet_level = (0.2 + random.random() * 0.4 * ema)),
        Reverb(room_size = (random.random() * ema), width = (random.random() * ema), damping = (random.random() * ema), dry_level = (0.2 + random.random() * 0.4 * ema))
    ]

    # Pedalboard creation
    num_pedals = random.randint(1,2)
    possible_pedals = [chorus_choices, compressor_choices, delay_choices, distortion_choices, gain_choices, phaser_choices, pitchshift_choices, reverb_choices]
    pedal_categories = np.random.choice(8, size=num_pedals, replace=False, p = [0.1, 0.1, 0.125, 0.175, 0.12, 0.105, 0.145, 0.13])
    if file_ending != "cln":
        board = []
        for i in pedal_categories:
            if possible_pedals[i] == distortion_choices or possible_pedals[i] == gain_choices:
                board.append(possible_pedals[i][0])
            else:
                board.append(random.choice(possible_pedals[i]))
    else:
        board = [random.choice(possible_pedals[i]) for i in pedal_categories]

    return Pedalboard(board)



def create_subtle_pedalboard(ema):
    """
    Function to create a pedalboard from the subtle_pedalboard object
    that has less extreme augmentations to be used on the interferer audio 
    samples in order to increase the variation of data the model sees.
    """

    subtle_pedalboard = [
        Reverb(room_size = (random.random() * 0.25 * ema), width = (random.random() * 0.25 * ema), damping = (random.random() * ema)),
        PitchShift(semitones =((1 if random.random() < 0.5 else -1) * random.random() * 5)),
        Phaser(rate_hz = (random.random() * 3 * ema), depth = (random.random() * 1.6 * ema), mix = (random.random() * 0.25 * ema)),
        Gain(gain_db = (random.random() * 10 * ema)),
        Distortion(drive_db = (random.random() * 15 * ema))
    ]

    num_pedals = random.choice([1, 2])
    pedal_categories = np.random.choice(5, size=num_pedals, replace=False)
    board = [subtle_pedalboard[i] for i in pedal_categories]

    return Pedalboard(board)