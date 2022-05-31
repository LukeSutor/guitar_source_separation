import os
import wave
import contextlib
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


PATH_TO_CLIPS = "C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/raw_clips"
SAVE_PATH = "C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/"
LENGTH = len(os.listdir(PATH_TO_CLIPS))

def minimizeFiles():

    for i, file in enumerate(os.listdir(PATH_TO_CLIPS)):

        snippets = file.split("|")

        for snippet in snippets:

            start_and_end = snippet.split("-")

            start_time = (60 * int(start_and_end[0].split(":")[0])) + (int(start_and_end[0].split(":")[1]))

            end_time = (60 * int(start_and_end[1].split(":")[0])) + (int(start_and_end[1].split(":")[1]))

            ffmpeg_extract_subclip(
                "C:/Users/Luke/Desktop/coding/solo_classifier/audio/full_solos/"+"/"+i+".wav",
                start_time,
                end_time,
                targetname=SAVE_PATH+(i + 1)+".wav"
            )