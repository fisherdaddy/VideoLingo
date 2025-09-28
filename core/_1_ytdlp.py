import os,sys
import glob
import re
import subprocess
from core.utils import *

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)

    def _get_optional_config(path):
        try:
            value = load_key(path)
        except KeyError:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best' if resolution == 'best' else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{
            'key': 'FFmpegThumbnailsConvertor',
            'format': 'jpg',
        }],
        'extractor_args': {
            'youtube': {
                # Try a couple of player clients; TV clients trigger fewer bot checks currently
                'player_client': ['tv_embedded', 'web'],
            }
        },
    }

    # Read Youtube Cookie File
    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()

    cookiefile = _get_optional_config('ytb_cookies.cookiefile')
    if cookiefile:
        expanded_cookiefile = os.path.abspath(os.path.expanduser(cookiefile))
        if not os.path.exists(expanded_cookiefile):
            print(f"Warning: Cookie file not found at {expanded_cookiefile}. Continuing without it.")
        else:
            ydl_opts['cookiefile'] = expanded_cookiefile

    browser = _get_optional_config('ytb_cookies.browser')
    if browser:
        profile = _get_optional_config('ytb_cookies.profile')
        cookies_args = [browser]
        if profile:
            cookies_args.append(profile)
        ydl_opts['cookiesfrombrowser'] = tuple(cookies_args)

    # Update yt-dlp to avoid download failure due to API changes
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to update yt-dlp: {e}")
    # Reload yt-dlp
    if 'yt_dlp' in sys.modules:
        del sys.modules['yt_dlp']
    from yt_dlp import YoutubeDL
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
