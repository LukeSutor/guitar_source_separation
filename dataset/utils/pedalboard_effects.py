import torch
import random
from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain, LadderFilter, Phaser
from pedalboard.io import AudioFile

# Read in a whole audio file: 'C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/00_Funk2-108-Eb_comp_hex_cln.wav'
with AudioFile('C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/guitar/00_Jazz2-110-Bb_comp_hex_cln.wav', 'r') as f:
    
    audio = f.read(f.frames)
    # print(audio)
    samplerate = f.samplerate

# Make a Pedalboard object, containing multiple plugins:
board = Pedalboard([
    Gain(gain_db=(35 * random.random)),
    Chorus(),
    Phaser(),
    Reverb(room_size=0.25),
])

# Run the audio through this pedalboard!
effected = board(audio, samplerate)
print(effected)
print(torch.tensor(effected))

# Write the audio back as a wav file:
with AudioFile('processed-output.wav', 'w', samplerate, effected.shape[0]) as f:
  f.write(effected)