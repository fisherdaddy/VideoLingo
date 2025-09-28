import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import whisperx
import torch
import time
import subprocess
from typing import Dict
from rich import print as rprint
import librosa
import tempfile
from core.utils.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import save_language

MODEL_DIR = load_key("model_dir")

# Global cache for models and mirrors to avoid repeated initialization
_cached_model = None
_cached_align_model = None
_cached_align_metadata = None
_cached_hf_mirror = None

def check_hf_mirror() -> str:
    """Check and return the fastest HF mirror - cached to avoid repeated checks"""
    global _cached_hf_mirror
    if _cached_hf_mirror:
        return _cached_hf_mirror
        
    mirrors = {
        'Official': 'huggingface.co',
        'Mirror': 'hf-mirror.com'
    }
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        try:
            if os.name == 'nt':
                cmd = ['ping', '-n', '1', '-w', '3000', domain]
            else:
                cmd = ['ping', '-c', '1', '-W', '3', domain]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            response_time = time.time() - start
            if result.returncode == 0:
                if response_time < best_time:
                    best_time = response_time
                    fastest_url = f"https://{domain}"
                rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
        except:
            rprint(f"[red]✗ {name}:[/red] Failed to connect")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    _cached_hf_mirror = fastest_url
    return fastest_url

def transcribe_audio(audio_file: str, start: float, end: float) -> Dict:
    global _cached_model, _cached_align_model, _cached_align_metadata
    
    # Memory monitoring for long videos
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            rprint(f"[yellow]⚠️ High memory usage: {memory_percent:.1f}%[/yellow]")
    except ImportError:
        pass
    
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Only print device info once for the first segment
    if _cached_model is None:
        rprint(f"🚀 Starting WhisperX using device: {device} ...")
        
        if device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            batch_size = 16 if gpu_mem > 8 else 2
            compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
            rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
        else:
            batch_size = 1
            compute_type = "int8"
            rprint(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        # Use cached batch size and compute type settings
        if device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            batch_size = 16 if gpu_mem > 8 else 2
            compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        else:
            batch_size = 1
            compute_type = "int8"
            
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    try:
        # Load main model (cached)
        if _cached_model is None:
            if WHISPER_LANGUAGE == 'zh':
                model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
                local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
            else:
                model_name = load_key("whisper.model")
                local_model = os.path.join(MODEL_DIR, model_name)
                
            if os.path.exists(local_model):
                rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
                model_name = local_model
            else:
                rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

            vad_options = {"vad_onset": 0.500,"vad_offset": 0.363}
            asr_options = {"temperatures": [0],"initial_prompt": "",}
            whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
            rprint("[bold yellow]**You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`**[/bold yellow]")
            _cached_model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

        # Create temp file with wav format for better compatibility
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio segment using ffmpeg
            ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 32000 -ac 1 "{temp_audio_path}"'
            subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
            
            # Load audio segment with librosa
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
            
            # Force garbage collection for long videos
            import gc
            gc.collect()
            
        finally:
            # Clean up temp file immediately
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")
        result = _cached_model.transcribe(audio_segment, batch_size=batch_size, print_progress=True)

        # Save language (only do this once)
        if _cached_align_model is None:
            save_language(result['language'])
            if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
                raise ValueError("Please specify the transcription language as zh and try again!")

            # Load align model (cached) with better error handling
            try:
                rprint(f"[cyan]🔗 Loading alignment model for language: {result['language']}...[/cyan]")
                
                # Special handling for Chinese - skip alignment if it causes issues
                if result['language'] == 'zh':
                    rprint("[yellow]⚠️ Chinese detected: Skipping word-level alignment to avoid hang issues[/yellow]")
                    rprint("[yellow]⚠️ Using segment-level timing only for better stability[/yellow]")
                    _cached_align_model = None
                    _cached_align_metadata = None
                else:
                    _cached_align_model, _cached_align_metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            except Exception as e:
                rprint(f"[yellow]⚠️ Failed to load alignment model: {e}[/yellow]")
                rprint("[yellow]⚠️ Continuing without word-level alignment...[/yellow]")
                _cached_align_model = None
                _cached_align_metadata = None

        # Align whisper output using cached model (with timeout protection)
        if _cached_align_model is not None:
            try:
                rprint("[cyan]🔗 Aligning transcript with audio...[/cyan]")
                result = whisperx.align(result["segments"], _cached_align_model, _cached_align_metadata, audio_segment, device, return_char_alignments=False)
            except Exception as e:
                rprint(f"[yellow]⚠️ Alignment failed: {e}[/yellow]")
                rprint("[yellow]⚠️ Using transcript without precise word-level timing...[/yellow]")
                # Continue with the original result without alignment

        # Adjust timestamps and standardize structure
        # Ensure segments list exists and each segment has optional 'words' list
        if not isinstance(result, dict) or 'segments' not in result:
            rprint("[yellow]⚠️ Unexpected transcription format; normalizing result[/yellow]")
            result = {'segments': result if isinstance(result, list) else []}

        for segment in result['segments']:
            # Normalize required keys
            segment['start'] = float(segment.get('start', 0.0)) + start
            segment['end'] = float(segment.get('end', 0.0)) + start
            # Some languages (e.g., zh) or failure to load align model
            # will produce segments without word-level timestamps.
            words = segment.get('words') or []
            for word in words:
                if 'start' in word:
                    word['start'] = float(word['start']) + start
                if 'end' in word:
                    word['end'] = float(word['end']) + start
            # Backfill missing 'words' with empty list to avoid KeyError downstream
            if 'words' not in segment:
                segment['words'] = []
        return result
    except Exception as e:
        rprint(f"[red]WhisperX processing error:[/red] {e}")
        raise

def cleanup_models():
    """Clean up cached models to free memory"""
    global _cached_model, _cached_align_model, _cached_align_metadata
    if _cached_model:
        del _cached_model
        _cached_model = None
    if _cached_align_model:
        del _cached_align_model
        _cached_align_model = None
    if _cached_align_metadata:
        _cached_align_metadata = None
    torch.cuda.empty_cache()
