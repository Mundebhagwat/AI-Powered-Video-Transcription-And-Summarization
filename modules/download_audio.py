import yt_dlp
import os

def download_audio_from_youtube(yt_url, output_path="data/output/video.%(ext)s"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': output_path,
        # 'ffmpeg_location': r'C:\ffmpeg\bin',
         'ffmpeg_location': os.getenv("FFMPEG_LOCATION", "/usr/bin"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])

    print(f"Audio downloaded and saved to {output_path}")
