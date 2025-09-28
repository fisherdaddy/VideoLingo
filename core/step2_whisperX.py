import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich import print as rprint
import subprocess

from core.utils.config_utils import load_key
from core.all_whisper_methods.demucs_vl import demucs_main, RAW_AUDIO_FILE, VOCAL_AUDIO_FILE
from core.all_whisper_methods.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, compress_audio, CLEANED_CHUNKS_EXCEL_PATH
from core.step1_ytdlp import find_video_files

WHISPER_FILE = "output/audio/for_whisper.mp3"
ENHANCED_VOCAL_PATH = "output/audio/enhanced_vocals.mp3"

def enhance_vocals(vocals_ratio=2.50):
    """Enhance vocals audio volume"""
    if not load_key("demucs"):
        return RAW_AUDIO_FILE
        
    try:
        print(f"[cyan]🎙️ Enhancing vocals with volume ratio: {vocals_ratio}[/cyan]")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{VOCAL_AUDIO_FILE}" '
            f'-filter:a "volume={vocals_ratio}" '
            f'"{ENHANCED_VOCAL_PATH}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        return ENHANCED_VOCAL_PATH
    except subprocess.CalledProcessError as e:
        print(f"[red]Error enhancing vocals: {str(e)}[/red]")
        return VOCAL_AUDIO_FILE  # Fallback to original vocals if enhancement fails
    
def transcribe(force: bool = False):
    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH) and not force:
        rprint("[yellow]⚠️ Transcription results already exist, skipping transcription step.[/yellow]")
        return
    
    # step0 Convert video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # step1 Demucs vocal separation (optional)
    if load_key("demucs"):
        try:
            demucs_main()
        except Exception as e:
            rprint(f"[yellow]⚠️ Demucs failed, falling back to raw audio: {e}[/yellow]")

    # step2 Choose audio to feed whisper
    if load_key("demucs") and os.path.exists(VOCAL_AUDIO_FILE):
        choose_audio = enhance_vocals()
    else:
        choose_audio = RAW_AUDIO_FILE
    # Compress audio for whisper
    whisper_audio = compress_audio(choose_audio, WHISPER_FILE)

    # step3 Extract audio
    segments = split_audio(whisper_audio)
    
    # step4 Transcribe audio
    all_results = []
    if load_key("whisper.runtime") == "local":
        from core.all_whisper_methods.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    else:
        from core.all_whisper_methods.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")

    for i, (start, end) in enumerate(segments, 1):
        rprint(f"[bold cyan]🎯 Processing segment {i}/{len(segments)} ({start/60:.1f}min - {end/60:.1f}min)[/bold cyan]")
        try:
            result = ts(whisper_audio, start, end)
        except Exception as e:
            # Don't abort entire run on a single segment failure
            rprint(f"[red]❌ Segment {i} failed: {e}[/red]")
            rprint("[yellow]⚠️ Skipping this segment and continuing...[/yellow]")
            result = {'segments': []}
        all_results.append(result)

        # Free memory between segments for long videos
        if len(segments) > 20:  # For videos with many segments
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    
    # Clean up models after processing all segments
    if load_key("whisper.runtime") == "local":
        from core.all_whisper_methods.whisperX_local import cleanup_models
        cleanup_models()
        rprint("[green]✅ Completed transcription and cleaned up models[/green]")
    
    # step5 Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # step6 Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()
