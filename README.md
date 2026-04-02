# MelodyWings Guard 🛡️

> **Real-Time Content Safety System for Neurodivergent Children**  
> A prototype that detects unsafe content across chat messages and video sessions.

---

## Overview

MelodyWings Guard is a multi-modal AI safety pipeline that scans:

| Channel | Checks |
|---------|--------|
| **Chat messages** | Profanity · PII (phone, email, address) · Toxicity (toxic-bert) |
| **Video frames** | NSFW classification (Falconsai) · Emotion detection (DeepFace) |
| **Video audio** | Whisper transcription → full chat safety pipeline |

All results are logged to `alerts_log.json` and visualised in a live Streamlit dashboard.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** must be installed and available on your PATH  
  - Windows: Install via [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) or `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

## Installation

### 1. Clone / navigate to the project folder

```bash
cd melodywings_guard
```

### 2. (Recommended) Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: The default install pulls the CPU version of PyTorch.  
> For GPU acceleration (CUDA), visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install the appropriate wheel before running the command above.

### 4. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 5. (Optional) First-run model caching

On first run, the following models are automatically downloaded and cached locally via HuggingFace Hub (no API key required):

| Model | Purpose | Approx. Size |
|-------|---------|--------------|
| `unitary/toxic-bert` | Toxicity classification | ~440 MB |
| `Falconsai/nsfw_image_detection` | NSFW image classification | ~90 MB |
| `openai/whisper-base` | Audio transcription | ~145 MB |

DeepFace models are also downloaded on first use (~100 MB total).

All models run **fully offline** after the initial download — **no API keys are needed**.

---

## Running the Safety Pipeline

```bash
python main.py
```

This will:
1. Analyse all `SAMPLE_MESSAGES` in `main.py` for profanity, PII, and toxicity
2. Analyse `test_video.mp4` (if present) for NSFW frames and dangerous emotions
3. Transcribe the video audio with Whisper and run the transcript through the chat pipeline
4. Log all results to `alerts_log.json`
5. Print a final summary report to the console

### Optional: test with a real video

Place any MP4 file in the same directory and rename it `test_video.mp4`, then re-run `main.py`.

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at [http://localhost:8501](http://localhost:8501) and:
- Display live KPI metrics (Total Scanned / Flagged / Chat / Video)
- Show a filterable alert log (filter by source, flagged-only toggle)
- Render an alert breakdown chart by reason type
- **Auto-refresh every 5 seconds** to pick up new alerts as `main.py` runs

---

## Project Structure

```
melodywings_guard/
├── main.py              # Entry point — orchestrates the full pipeline
├── chat_analyzer.py     # Chat safety: profanity, PII, toxicity
├── video_analyzer.py    # Video safety: NSFW frames, emotion, transcript
├── alert_engine.py      # Unified alert logger + alerts_log.json persistence
├── dashboard.py         # Streamlit real-time dashboard
├── requirements.txt     # All Python dependencies
└── README.md            # This file
```

Generated at runtime:
```
melodywings_guard/
├── alerts_log.json          # Persisted alert records (JSON array)
└── melodywings_guard.log    # Detailed Python logging output
```

---

## Architecture

```
SAMPLE_MESSAGES ──► chat_analyzer.py ──────────────────┐
                      ├── better-profanity              │
                      ├── Regex PII detection           │
                      └── HuggingFace toxic-bert        │
                                                        ▼
test_video.mp4 ───► video_analyzer.py          alert_engine.py
                      ├── OpenCV frame extract  ├── Adds "source" field
                      ├── Falconsai NSFW        ├── Appends to alerts_log.json
                      ├── DeepFace emotion      └── Prints summary
                      ├── ffmpeg audio extract          │
                      └── Whisper transcription         │
                            └── chat_analyzer           │
                                                        ▼
                                                 dashboard.py
                                                 (Streamlit UI)
```

---

## Safety Thresholds (configurable)

| Check | Default Threshold | Location |
|-------|------------------|----------|
| Toxicity flagging | ≥ 0.75 confidence | `chat_analyzer.py` → `TOXICITY_THRESHOLD` |
| NSFW flagging | ≥ 0.70 confidence | `video_analyzer.py` → `NSFW_THRESHOLD` |
| Flagged emotions | fear, disgust, sad, angry | `video_analyzer.py` → `FLAGGED_EMOTIONS` |

---

## Offline / Privacy Notes

- **No data leaves your machine.** All models run locally.
- `alerts_log.json` is stored locally in the project folder.
- Video audio is extracted to a temporary directory and automatically deleted after transcription.
- The system is designed as a **prototype** — not a production deployment. For production, add authentication, rate limiting, and data retention policies.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg and add to PATH (see Prerequisites) |
| `CUDA out of memory` | Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` or use CPU |
| DeepFace download error | Check internet connection on first run; models are cached after that |
| `ModuleNotFoundError: tf_keras` | Run `pip install tf-keras` |
| Dashboard shows no data | Run `python main.py` first to populate `alerts_log.json` |
