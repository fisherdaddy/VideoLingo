# Repository Guidelines

## Project Structure & Module Organization
Core pipeline stages live in `core/` and follow the `stepN_<verb_noun>.py` naming (e.g., `core/step4_2_translate_all.py`). Streamlit UI entrypoint is `st.py`, with reusable widgets and utilities in `st_components/`. Shared assets sit under `docs/`, while localized strings are in `translations/`. Generated artifacts land in `output/` and `history/`; `_model_cache/` stores large models and stays ignored. Batch automations reside in `batch/` alongside `tasks_setting.xlsx` for bulk runs. Keep new modules small, single-purpose, and explicitly imported where used.

## Build, Test, and Development Commands
Run `python install.py` to set up dependencies, auto-detect GPU, and validate ffmpeg before launching the UI. Start local development with `streamlit run st.py`. For containerized workflows, `docker build -t videolingo .` then `docker run -d -p 8501:8501 --gpus all videolingo`. Windows batch processing uses `batch/OneKeyBatch.bat` after configuring `batch/tasks_setting.xlsx`.

## Coding Style & Naming Conventions
Target Python 3.10 with PEP 8 formatting and 4-space indentation. Use snake_case for variables and functions, PascalCase for classes, and keep pipeline step names aligned with their responsibility. Prefer explicit imports from `core` modules. Add comments sparingly to clarify non-obvious logic. When expanding the pipeline, follow the established step sequence and wire the module through `st_components/imports_and_utils.py` and `st.py`.

## Testing Guidelines
No formal unit suite yet; perform smoke tests by processing a short sample video. Confirm `output/*.srt`, `output/output_sub.mp4`, and/or `output/output_dub.mp4` are produced and that `output/log/` entries look sane. If contributing automated tests, use `pytest`, name files `tests/test_<module>.py`, and isolate file I/O via temporary directories.

## Commit & Pull Request Guidelines
Use Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) with ≤72-character subjects and optional scope for clarity. PRs should summarize intent, link issues, and include before/after evidence (logs, screenshots, short clips) when the behavior changes. Highlight any config or migration impacts and keep changes focused.

## Security & Configuration Tips
Never commit real API keys; rely on placeholders in `config.yaml` or environment variables. Double-check generated media for sensitive content before sharing. Large models stay under `_model_cache/`; avoid adding binaries or cached artifacts to version control.
