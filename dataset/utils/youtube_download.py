from __future__ import unicode_literals
import youtube_dl


videos = [
    "https://www.youtube.com/watch?v=E6jyB6qUuzg - 1.23-4.39&4.40-8.15",
    "https://www.youtube.com/watch?v=wgRh7AYyLO8 - 0.52-2.38&2.39-5.28",
    "https://www.youtube.com/watch?v=EacRWe797Hk - 1.08-3.20&3.21-6.26",
    "https://www.youtube.com/watch?v=1Z0-dq1Lvak - 0.01-3.36&3.37-7.22&7.23-9.56",
    "https://www.youtube.com/watch?v=t5q0TSI1c9c - 1.16-4.47&4.48-9.02",
    "https://www.youtube.com/watch?v=ekHSIXxCjUA - 1.35-4.53&4.54-8.52&8.53-11.40",
    "https://www.youtube.com/watch?v=kqL8bZAC5T0 - 1.01-3.20&3.21-6.31&6.32-7.46",
    "https://www.youtube.com/watch?v=ROTksl2bdG8 - 1.59-6.15&6.16-11.32&11.33-14.38",
    "https://www.youtube.com/watch?v=eINkbX_HpnI - 0.59-5.07&5.08-8.11",
    "https://www.youtube.com/watch?v=xC8HsHE-3fE - 1.06-4.35&4.36-6.44",
    "https://www.youtube.com/watch?v=ML-DSvQSa2E - 1.03-4.12&4.14-6.30",
    "https://www.youtube.com/watch?v=rVV1pmXL__0 - 0.28-2.32&2.33-5.20",
    "https://www.youtube.com/watch?v=xm1dPof-HcY - 1.01-3.21&3.22-5.25",
    "https://www.youtube.com/watch?v=g-YdVAWsdo0 - 1.06-5.19&5.22-7.53",
    "https://www.youtube.com/watch?v=rgdkMHyYNuw - 1.07-3.58&3.59-7.13",
    "https://www.youtube.com/watch?v=H1ovmUIxB-g - 0.39-3.10&3.12-5.45",
    "https://www.youtube.com/watch?v=iWMAY9mqElQ - 0.40-4.13&4.14-7.33",
    "https://www.youtube.com/watch?v=LYo9gwNntwA - 0.01-2.32&2.33-4.40",
    "https://www.youtube.com/watch?v=nVY2h-uptps - 1.06-4.16&4.19-8.19",
    "https://www.youtube.com/watch?v=6y9ODn3BnWY - 0.59-3.34&3.39-5.12",
    "https://www.youtube.com/watch?v=q1A4yw9ABjg - 0.01-2.58&2.59-5.34",

]


if __name__ == "__main__":
    for video in videos:

        url = video.split(" - ")[0]
        times = video.split(" - ")[1]

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': '../youtube_clips/%(title)s. - '+times+'.%(ext)s',
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print("error downloading "+url)
            print(e)
            continue

    print("fin.")
