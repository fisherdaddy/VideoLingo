# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
- `python install.py` - Interactive installer that handles PyPI mirrors, PyTorch (GPU/CPU), requirements, and starts the app
- `conda create -n videolingo python=3.10.0 -y && conda activate videolingo` - Create and activate conda environment
- `streamlit run st.py` - Start the Streamlit web interface

### Running the Application
- **Main Interface**: `streamlit run st.py` - Web-based UI for video translation and dubbing
- **Batch Processing**: `python batch/utils/batch_processor.py` or run `OneKeyBatch.bat` (Windows)

### Testing Individual Components
Each processing step can be run independently:
- `python core/step1_ytdlp.py` - YouTube video download
- `python core/step2_whisperX.py` - Audio transcription with WhisperX  
- `python core/step3_1_spacy_split.py` - NLP-based sentence segmentation
- `python core/step4_1_summarize.py` - Content summarization
- `python core/step4_2_translate_all.py` - Multi-step translation
- `python core/step5_splitforsub.py` - Subtitle splitting and alignment
- `python core/step10_gen_audio.py` - TTS audio generation
- `python core/step12_merge_dub_to_vid.py` - Final video dubbing

### Configuration
- `config.yaml` - Main configuration file for API keys, language settings, TTS options
- `custom_terms.xlsx` - Custom terminology dictionary for translation consistency
- `batch/tasks_setting.xlsx` - Batch processing task configuration

## Architecture Overview

VideoLingo is a comprehensive video translation and dubbing pipeline with these key components:

### Core Processing Pipeline
The application follows a 12-step sequential processing pipeline:

1. **Video Input** (`step1_ytdlp.py`) - YouTube download or local video processing
2. **Audio Extraction & Preprocessing** (`all_whisper_methods/`) - Audio separation with Demucs, WhisperX transcription
3. **Text Processing** (`step3_*.py`) - NLP segmentation using spaCy and LLM-based meaning splits  
4. **Translation** (`step4_*.py`) - Multi-step translation with summarization, terminology, and reflection
5. **Subtitle Generation** (`step5_splitforsub.py`, `step6_generate_final_timeline.py`) - Timeline alignment and subtitle formatting
6. **Video Merging** (`step7_merge_sub_to_vid.py`) - Subtitle embedding
7. **Audio Generation** (`step8_*.py`, `step9_*.py`, `step10_gen_audio.py`) - TTS processing with multiple providers
8. **Audio Processing** (`step11_merge_full_audio.py`) - Audio timeline merging
9. **Final Output** (`step12_merge_dub_to_vid.py`) - Complete dubbed video generation

### Key Modules

- **`core/ask_gpt.py`** - Centralized LLM API interface supporting OpenAI-compatible APIs
- **`core/config_utils.py`** - Configuration management with yaml file handling
- **`core/all_tts_functions/`** - TTS provider implementations (Azure, OpenAI, Edge, GPT-SoVITS, etc.)
- **`core/all_whisper_methods/`** - WhisperX integration for word-level transcription
- **`core/spacy_utils/`** - NLP utilities for intelligent sentence segmentation
- **`st_components/`** - Streamlit UI components
- **`batch/`** - Batch processing system for multiple videos

### Configuration Structure
The `config.yaml` file contains:
- API credentials for LLM, TTS, and transcription services
- Language detection and target language settings  
- Processing parameters (subtitle length, speed factors, etc.)
- TTS voice configurations for different providers
- Advanced settings for subtitle alignment and audio merging

### Data Flow
- **Input**: Video files (local or YouTube URLs)
- **Intermediate**: `output/` directory with processing logs, audio files, subtitle files
- **Output**: Subtitled videos (`output_sub.mp4`) and dubbed videos (`output_dub.mp4`)
- **Logs**: Detailed processing logs in `output/gpt_log/` and `output/log/`

### Multi-language Support
- UI translations in `translations/` with JSON files for different languages
- spaCy model integration for multiple languages (configured in `config.yaml`)
- Language-specific processing rules for subtitle formatting

### Batch Processing
- Excel-based task configuration (`batch/tasks_setting.xlsx`)
- Parallel processing capabilities with error handling
- Status tracking and recovery mechanisms for failed tasks

## Development Notes

- The application uses a modular step-by-step architecture that allows resuming from any point in the pipeline
- Configuration is centralized but can be overridden per-step for flexibility
- All LLM interactions go through the `ask_gpt.py` abstraction layer
- TTS providers are pluggable through the `all_tts_functions/` module system
- The Streamlit interface provides real-time progress tracking and parameter adjustment

## Dependencies

- **Core**: Python 3.10, PyTorch, Streamlit, WhisperX, spaCy
- **Audio**: librosa, pydub, demucs, moviepy  
- **AI/ML**: transformers, openai, various TTS provider SDKs
- **Utilities**: pandas, openpyxl, yt-dlp, json-repair