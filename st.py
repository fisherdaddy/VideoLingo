import streamlit as st
import os, sys
from core.st_utils.imports_and_utils import *
from core import *
# Reduce torchaudio deprecation warnings in logs
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "1")
from core.utils.config_utils import load_key

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
ASR_TIMELINE_EXCEL = "output/log/cleaned_chunks.xlsx"
ASR_SRT_PATH = "output/asr_only.srt"

def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """, unsafe_allow_html=True)

        if not os.path.exists(SUB_VIDEO):
            if st.button(t("Start Processing Subtitles"), key="text_processing_button"):
                process_text()
                st.rerun()
        else:
            if load_key("burn_subtitles"):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))
            
            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True

def process_text():
    with st.spinner(t("Using Whisper for transcription...")):
        _2_asr.transcribe()
    with st.spinner(t("Splitting long sentences...")):  
        _3_1_split_nlp.split_by_spacy()
        _3_2_split_meaning.split_sentences_by_meaning()
    with st.spinner(t("Summarizing and translating...")):
        _4_1_summarize.get_summary()
        if load_key("pause_before_translate"):
            input(t("⚠️ PAUSE_BEFORE_TRANSLATE. Go to `output/log/terminology.json` to edit terminology. Then press ENTER to continue..."))
        _4_2_translate.translate_all()
    with st.spinner(t("Processing and aligning subtitles...")): 
        _5_split_sub.split_for_sub_main()
        _6_gen_sub.align_timestamp_main()
    with st.spinner(t("Merging subtitles to video...")):
        _7_sub_into_vid.merge_subtitles_to_video()
    
    st.success(t("Subtitle processing complete! 🎉"))
    st.balloons()

def audio_processing_section():
    st.header(t("c. Dubbing"))
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """, unsafe_allow_html=True)
        if not os.path.exists(DUB_VIDEO):
            if st.button(t("Start Audio Processing"), key="audio_processing_button"):
                process_audio()
                st.rerun()
        else:
            st.success(t("Audio processing is complete! You can check the audio files in the `output` folder."))
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO) 
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()

def process_audio():
    with st.spinner(t("Generate audio tasks")): 
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
    with st.spinner(t("Extract refer audio")):
        _9_refer_audio.extract_refer_audio_main()
    with st.spinner(t("Generate all audio")):
        _10_gen_audio.gen_audio()
    with st.spinner(t("Merge full audio")):
        _11_merge_audio.merge_full_audio()
    with st.spinner(t("Merge dubbing to the video")):
        _12_dub_to_vid.merge_video_audio()
    
    st.success(t("Audio processing complete! 🎇"))
    st.balloons()

def asr_only_section():
    st.header(t("a. 仅进行 ASR 识别"))
    with st.container(border=True):
        st.markdown(t("仅对原视频进行 WhisperX 识别，导出标准 SRT 字幕文件；不进行分句、翻译或配音。识别完成后会显示完成标记与下载按钮。"))

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("开始仅 ASR 识别"), key="asr_only_button_start"):
                with st.spinner(t("正在进行语音识别，请稍候…")):
                    step2_whisperX.transcribe(force=True)
                st.success(t("识别完成！可下载 SRT 文件。"))
                st.rerun()
        with col2:
            if st.button(t("重新执行 ASR"), key="asr_only_button_rerun"):
                with st.spinner(t("正在重新识别，请稍候…")):
                    step2_whisperX.transcribe(force=True)
                st.success(t("已重新识别！可下载最新 SRT 文件。"))
                st.rerun()

        if os.path.exists(ASR_SRT_PATH):
            with open(ASR_SRT_PATH, "rb") as f:
                srt_bytes = f.read()
            # 状态与统计
            try:
                import datetime
                mtime = os.path.getmtime(ASR_SRT_PATH)
                ts_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                # 估算字幕条数（以空行分割）
                txt = srt_bytes.decode('utf-8', errors='ignore')
                cues = [blk for blk in txt.strip().split('\n\n') if blk.strip()]
                st.success(t(f"ASR 已完成，共 {len(cues)} 条字幕，完成时间：{ts_str}"))
            except Exception:
                st.success(t("ASR 已完成，可下载 SRT 文件。"))

            st.download_button(
                label=t("下载 ASR 时间轴（.srt）"),
                data=srt_bytes,
                file_name="asr_timeline.srt",
                mime="application/x-subrip",
            )
        else:
            st.info(t("暂无可下载的 SRT 文件，请先执行 ASR。"))

def main():
    logo_col, _ = st.columns([1,1])
    with logo_col:
        st.image("docs/logo.png", use_column_width=True)
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t("Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href=\"https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh\" target=\"_blank\">here</a>! You can also try out our SaaS website at <a href=\"https://videolingo.io\" target=\"_blank\">videolingo.io</a> for free!")
    st.markdown(f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>", unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    download_video_section()
    asr_only_section()
    text_processing_section()
    audio_processing_section()

if __name__ == "__main__":
    main()
