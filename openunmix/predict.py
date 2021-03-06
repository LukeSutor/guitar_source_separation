import torch
import torchaudio
import utils

DEVICE = "cpu"
SAMPLE_RATE = 44100

def separate(
    audio,
    rate=None,
    model_str_or_path="umxl",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None,
    filterbank="torch",
    source="pth"
):
    """
    Open Unmix functional interface

    Separates a torch.Tensor or the content of an audio file.

    If a separator is provided, use it for inference. If not, create one
    and use it afterwards.

    Args:
        audio: audio to process
            torch Tensor: shape (channels, length), and
            `rate` must also be provided.
        rate: int or None: only used if audio is a Tensor. Otherwise,
            inferred from the file.
        model_str_or_path: the pretrained model to use, defaults to UMX-L
        targets (str): select the targets for the source to be separated.
            a list including: ['vocals', 'drums', 'bass', 'other'].
            If you don't pick them all, you probably want to
            activate the `residual=True` option.
            Defaults to all available targets per model.
        niter (int): the number of post-processingiterations, defaults to 1
        residual (bool): if True, a "garbage" target is created
        wiener_win_len (int): the number of frames to use when batching
            the post-processing step
        aggregate_dict (str): if provided, must be a string containing a '
            'valid expression for a dictionary, with keys as output '
            'target names, and values a list of targets that are used to '
            'build it. For instance: \'{\"vocals\":[\"vocals\"], '
            '\"accompaniment\":[\"drums\",\"bass\",\"other\"]}\'
        separator: if provided, the model.Separator object that will be used
             to perform separation
        device (str): selects device to be used for inference
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
            filterbank=filterbank,
            source=source
        )
        separator.freeze()
        if device:
            separator.to(device)

    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)
    return estimates



def Transform(audio_dir, device=DEVICE):
    signal, sr = torchaudio.load(audio_dir)

    signal = signal.to(device)
    signal = _mix_down_if_necessary(signal)
    signal = _resample_if_necessary(signal, sr, device)

    return signal, SAMPLE_RATE


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
    root_dir = "../dataset/data/test/"
    filename = "mississippi_queen"
    file = root_dir + filename + ".wav"
    model_path = "./open-unmix"
    save_dir = "../dataset/data/separations/"
    source = "chkpnt"
    audio, rate = Transform(file)
    estimates = separate(audio, rate, model_path, "guitar", residual=True, source=source)
    torchaudio.save(save_dir+filename+"_guitar.wav", estimates['guitar'][0], SAMPLE_RATE)
    torchaudio.save(save_dir+filename+"_residual.wav", estimates['residual'][0], SAMPLE_RATE)
