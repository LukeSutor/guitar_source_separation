import dis
import random
import numpy as np
import torch
import time
from pedalboard import Pedalboard, Chorus, Reverb, Compressor, Gain, Phaser, Delay, Distortion, PitchShift
from pedalboard.io import AudioFile

# with AudioFile('../data/guitar/00_BN2-131-B_solo_hex_cln.wav', 'r') as f:
#   audio = f.read(f.frames)
#   samplerate = f.samplerate


chorus_choices = [
    Chorus(centre_delay_ms = (random.choice([7,8])), depth = (0.10 + random.random() * 0.25), feedback = (0.10 + random.random() * 0.25)), # Classic chorus
    Chorus(centre_delay_ms = (random.choice([1,2])), depth = (random.random() * 0.15), feedback=(0.7 + random.random() * 0.25)), # Flanger
    Chorus(centre_delay_ms = (random.choice([1,2])), depth=(random.random() * 0.15), feedback=(0.7 + random.random() * 0.25), mix=1) # Vibrato
]

compressor_choices = [
    Compressor()
]

delay_choices = [
    Delay(delay_seconds = (random.random() * 0.5)),
    Delay(delay_seconds = (random.random() * 0.5), feedback=(0.1 + random.random() * 0.15)),
    Delay(delay_seconds = (random.random() * 0.5), mix = (0.35 + random.random() * 0.4)),
    Delay(delay_seconds = (random.random() * 0.5), feedback=(0.1 + random.random() * 0.15), mix = (0.35 + random.random() * 0.4)),
]

distortion_choices = [
    Distortion(drive_db = (25 + random.random() * 35)), # High
    Distortion(drive_db = (random.random() * 25)) # Low
]

gain_choices = [
    Gain(gain_db = (15 + random.random() * 15)), # High
    Gain(gain_db = (random.random() * 15)) # Low
]

phaser_choices = [
    Phaser(rate_hz = (random.random()), depth = (0.8 + random.random() * 0.8)),
    Phaser(rate_hz = (1 + random.random() * 2), depth = (random.random() * 0.8)),
    Phaser(rate_hz = (random.random()), depth = (0.8 + random.random() * 0.8), feedback = (0.2 + random.random() * 0.5)),
    Phaser(rate_hz = (1 + random.random() * 2), depth = (random.random() * 0.8), feedback = (0.2 + random.random() * 0.5)),
]

pitchshift_choices = [
    PitchShift(semitones =(random.random() * 5)), #Higher pitch
    PitchShift(semitones =(-random.random() * 5)) # Lower pitch
]
 
reverb_choices = [
    Reverb(room_size = (random.random()), width = (random.random()), damping = (random.random())),
    Reverb(room_size = (random.random()), width = (random.random()), damping = (random.random()), wet_level = (0.2 + random.random() * 0.4)),
    Reverb(room_size = (random.random()), width = (random.random()), damping = (random.random()), dry_level = (0.2 + random.random() * 0.4))
]

def create_pedalboard(file_ending):
    num_pedals = random.randint(1,4)
    possible_pedals = [chorus_choices, compressor_choices, delay_choices, distortion_choices, gain_choices, phaser_choices, pitchshift_choices, reverb_choices]
    pedal_categories = np.random.choice(8, size=num_pedals, replace=False)
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


subtle_pedalboard = [
    Reverb(room_size = (random.random() * 0.25), width = (random.random() * 0.25), damping = (random.random())),
    PitchShift(semitones =((1 if random.random() < 0.5 else -1) * random.random() * 5)),
    Phaser(rate_hz = (random.random() * 3), depth = (random.random() * 1.6), mix = (random.random() * 0.25)),
    Gain(gain_db = (random.random() * 10)),
    Distortion(drive_db = (random.random() * 15))
]

def create_subtle_pedalboard():
    num_pedals = random.choice([1,2,3])
    pedal_categories = np.random.choice(5, size=num_pedals, replace=False)
    board = [subtle_pedalboard[i] for i in pedal_categories]

    return Pedalboard(board)
                

# board = create_pedalboard('clna')

# Run the audio through this pedalboard!
# effected = board(audio, samplerate)

# Write the audio back as a wav file:
# with AudioFile('processed-output.wav', 'w', samplerate, effected.shape[0]) as f:
#   f.write(effected)

frame_offset = 10000

num_frames = 264600

start = time.time()
for i in range(100):
    with AudioFile('../data/guitar/00_BN2-131-B_solo_hex_cln.wav', 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    board = create_pedalboard('cln')
    effected = board(audio, samplerate)
    array = np.array([effected[0][frame_offset:frame_offset+num_frames]])
    sig = torch.from_numpy(array)

end = time.time()

print("total:",end-start,"seconds")