import os
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


PATH_TO_CLIPS = "C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/youtube_clips/"
SAVE_PATH = "C:/Users/Luke/Desktop/coding/unmix_guitar_separation/dataset/data/clipped_youtube/"

def minimizeFiles():

    total_time = 0

    for file in os.listdir(PATH_TO_CLIPS):

        if file.endswith('.wav'):

            segments = file[:-4].split(' - ')

            snippets = segments[-1].split("&")


            for j, snippet in enumerate(snippets):

                start_and_end = snippet.split("-")

                start_time = (60 * int(start_and_end[0].split(".")[0])) + (int(start_and_end[0].split(".")[1]))

                end_time = (60 * int(start_and_end[1].split(".")[0])) + (int(start_and_end[1].split(".")[1]))

                total_time += end_time - start_time

                ffmpeg_extract_subclip(
                    PATH_TO_CLIPS+file,
                    start_time,
                    end_time,
                    targetname=SAVE_PATH+' - '.join(segments[:-1])+'_'+str(j + 1)+".wav"
                ) 

    print("saved",math.floor(total_time / 36) / 100,"hours of clips.")


if __name__ == "__main__":
    minimizeFiles()