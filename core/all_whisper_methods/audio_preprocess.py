import os, sys, subprocess, re
import pandas as pd
from typing import Dict, List, Tuple
from rich import print
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.utils.config_utils import update_key, load_key

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = "output/audio/raw.mp3"
CLEANED_CHUNKS_EXCEL_PATH = "output/log/cleaned_chunks.xlsx"
ASR_SRT_PATH = "output/asr_only.srt"

def _seconds_to_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _clean_srt_text(text: str) -> str:
    # Remove spaces between consecutive CJK characters and around CJK punctuation
    text = re.sub(r'(?<=[\u4e00-\u9FFF])\s+(?=[\u4e00-\u9FFF])', '', text)
    text = re.sub(r'\s*([，。！？、；：“”‘’（）《》])\s*', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def _words_to_srt_segments(df: pd.DataFrame, gap_threshold: float = 0.6, max_chars: int = 70) -> List[Tuple[float, float, str]]:
    segs: List[Tuple[float, float, str]] = []
    if df.empty:
        return segs
    buffer_words: List[str] = []
    seg_start = float(df.iloc[0]['start'])
    last_end = float(df.iloc[0]['end'])
    for idx, row in df.iterrows():
        w = str(row['text'])
        start = float(row['start'])
        end = float(row['end'])
        gap = start - last_end
        # decide if new segment
        candidate = " ".join(buffer_words + [w]) if buffer_words else w
        candidate_clean = _clean_srt_text(candidate)
        if buffer_words and (gap > gap_threshold or len(candidate_clean) > max_chars):
            seg_text = _clean_srt_text(" ".join(buffer_words))
            segs.append((seg_start, last_end, seg_text))
            buffer_words = [w]
            seg_start = start
        else:
            buffer_words.append(w)
        last_end = end
    if buffer_words:
        seg_text = _clean_srt_text(" ".join(buffer_words))
        segs.append((seg_start, last_end, seg_text))
    return segs

def save_asr_srt(df: pd.DataFrame, srt_path: str = ASR_SRT_PATH):
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    # Ensure proper order and text without quoting
    df_sorted = df.sort_values(by=['start', 'end']).copy()
    # build segments and write srt
    segments = _words_to_srt_segments(df_sorted)
    lines = []
    for i, (st_sec, en_sec, txt) in enumerate(segments, start=1):
        start_srt = _seconds_to_srt_ts(st_sec)
        end_srt = _seconds_to_srt_ts(en_sec)
        lines.append(f"{i}\n{start_srt} --> {end_srt}\n{txt}\n")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"📜 SRT file saved to {srt_path}")

def compress_audio(input_file: str, output_file: str):
    """将输入音频文件压缩为低质量音频文件，用于转录"""
    if not os.path.exists(output_file):
        print(f"🗜️ Converting to low quality audio with FFmpeg ......")
        # 16000 Hz, 1 channel, (Whisper default) , 96kbps to keep more details as well as smaller file size
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file, '-vn', '-b:a', '96k',
            '-ar', '16000', '-ac', '1', '-metadata', 'encoding=UTF-8',
            '-f', 'mp3', output_file
        ], check=True, stderr=subprocess.PIPE)
        print(f"🗜️ Converted <{input_file}> to <{output_file}> with FFmpeg")
    return output_file

def convert_video_to_audio(video_file: str):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if not os.path.exists(RAW_AUDIO_FILE):
        print(f"🎬➡️🎵 Converting to high quality audio with FFmpeg ......")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '32000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        print(f"🎬➡️🎵 Converted <{video_file}> to <{RAW_AUDIO_FILE}> with FFmpeg\n")

def _detect_silence(audio_file: str, start: float, end: float) -> List[float]:
    """Detect silence points in the given audio segment"""
    cmd = ['ffmpeg', '-y', '-i', audio_file, 
           '-ss', str(start), '-to', str(end),
           '-af', 'silencedetect=n=-30dB:d=0.5', 
           '-f', 'null', '-']
    
    output = subprocess.run(cmd, capture_output=True, text=True, 
                          encoding='utf-8').stderr
    
    return [float(line.split('silence_end: ')[1].split(' ')[0])
            for line in output.split('\n')
            if 'silence_end' in line]

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: int | None = None, win: int | None = None) -> List[Tuple[float, float]]:
    """Split audio into segments near a target length, preferring silence boundaries.

    Configurable via config.yaml under keys:
      - whisper.runtime: 'local' | 'cloud'
      - whisper.segment.max_seconds_local: int (default 1200)
      - whisper.segment.max_seconds_cloud: int (default 600)
      - whisper.segment.silence_window: int seconds around boundary to search (default 30)

    Rationale: WhisperX internally chunks audio; larger external segments reduce overhead
    without increasing peak memory significantly. Defaults aim for larger, stable chunks.
    """
    print("[bold blue]🔪 Starting audio segmentation...[/]")

    # Read duration once
    duration = get_audio_duration(audio_file)
    print(f"[cyan]📏 Total audio duration: {duration/3600:.1f} hours ({duration:.1f} seconds)[/cyan]")

    # Load config with safe defaults
    def cfg(key: str, default):
        try:
            return load_key(key)
        except Exception:
            return default

    runtime = cfg("whisper.runtime", "local")
    default_target = cfg(
        "whisper.segment.max_seconds_local" if runtime == "local" else "whisper.segment.max_seconds_cloud",
        1200 if runtime == "local" else 600,
    )
    target_len = target_len or int(default_target)
    win = int(win or cfg("whisper.segment.silence_window", 30))

    # Clamp to sensible bounds
    min_seconds = int(cfg("whisper.segment.min_seconds", 120))
    max_seconds = int(cfg("whisper.segment.max_seconds", 1800))
    target_len = max(min_seconds, min(target_len, max_seconds))

    # Build segments preferring silence near each boundary
    segments: List[Tuple[float, float]] = []
    pos = 0.0
    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration))
            break

        # Search for a silence close to the target boundary
        win_start = max(0.0, pos + target_len - win)
        win_end = min(win_start + 2 * win, duration)
        silences = _detect_silence(audio_file, win_start, win_end)

        if silences:
            # Find the first silence after the intended boundary within the window
            target_pos = target_len - (win_start - pos)
            split_at = next((t for t in silences if t - win_start > target_pos), None)
            if split_at:
                segments.append((pos, split_at))
                pos = split_at
                continue

        # Fallback: hard split at target boundary if no silence found
        hard_end = min(pos + target_len, duration)
        segments.append((pos, hard_end))
        pos = hard_end

    print(f"🔪 Audio split into {len(segments)} segments (≈{target_len//60} min each)")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Handle cases where there might not be word-level alignment
        if 'words' not in segment or not segment['words']:
            # If no word-level data, create pseudo-words from segment text
            segment_text = segment.get('text', '').strip()
            if segment_text:
                # For Chinese and other languages without word-level alignment
                # Split by common punctuation or spaces
                import re
                words = re.split(r'([，。！？；：、\s]+)', segment_text)
                words = [w.strip() for w in words if w.strip()]
                
                # Distribute timing evenly across words
                segment_duration = segment['end'] - segment['start']
                word_duration = segment_duration / len(words) if words else 0
                
                for i, word_text in enumerate(words):
                    word_start = segment['start'] + (i * word_duration)
                    word_end = segment['start'] + ((i + 1) * word_duration)
                    
                    word_dict = {
                        'text': word_text,
                        'start': word_start,
                        'end': word_end,
                    }
                    all_words.append(word_dict)
            continue
            
        # Original word-level processing
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 20:
                print(f"⚠️ Warning: Detected word longer than 20 characters, skipping: {word['word']}")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                        }
                        all_words.append(word_dict)
                    else:
                        # Fallback: use segment timing
                        word_dict = {
                            'text': word["word"],
                            'start': segment['start'],
                            'end': segment['end'],
                        }
                        all_words.append(word_dict)
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word.get('start', all_words[-1]['end'] if all_words else segment['start']),
                    'end': word.get('end', segment['end']),
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)
    # Save SRT before mutating text for Excel
    try:
        save_asr_srt(df.copy())
    except Exception as e:
        print(f"[yellow]⚠️ Failed to save SRT: {e}[/yellow]")

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"ℹ️ Removed {removed_rows} row(s) with empty text.")
    
    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 20]
    if not long_words.empty:
        print(f"⚠️ Warning: Detected {len(long_words)} word(s) longer than 20 characters. These will be removed.")
        df = df[df['text'].str.len() <= 20]
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(CLEANED_CHUNKS_EXCEL_PATH, index=False)
    print(f"📊 Excel file saved to {CLEANED_CHUNKS_EXCEL_PATH}")

def save_language(language: str):
    update_key("whisper.detected_language", language)
