# MelodyWings Guard

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/AI%2FML-Transformers%20%7C%20DeepFace-FF6F61" alt="AI/ML" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Backend-Flask-0E1117" alt="Backend" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Database-SQLite%20%7C%20PostgreSQL-336791" alt="Database" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Production%20Ready%20Prototype-2EA44F" alt="Status" /></a>
</p>

> **Multimodal content safety platform** that analyzes **video, audio, and text end-to-end**.  
> Transforms raw media into structured moderation intelligence through a sophisticated pipeline of computer vision, speech recognition, NLP, alert scoring, and interactive dashboards.

---

## 🎯 Project Overview

MelodyWings Guard is a **production-ready** safety analysis system designed to identify and flag risky content across multiple modalities. It processes video uploads and simultaneously:

- **📹 Analyzes visual frames** for NSFW content and emotional signals
- **🎙️ Extracts and analyzes audio** for behavioral anomalies (volume, silence, speech patterns)
- **🗣️ Transcribes speech to text** using best-effort ASR (Whisper + fallback)
- **💬 Applies NLP safety checks** (toxicity, profanity, PII, sentiment, entities)
- **⚠️ Generates unified alerts** with severity-based scoring and confidence metadata
- **📊 Powers the HTML dashboard** for investigation, analytics, and export workflows
- **Upload & Review page** for live upload, progress, transcript review, flagged frames, and validation

---

## 🏗️ System Architecture (Detailed)

### End-to-End Data Flow Diagram

```mermaid
graph TD
    subgraph Input["📥 Input Layer"]
        VID["Video File<br/>(MP4, MKV, etc.)"] 
    end

    subgraph Extract["🎬 Extraction Stage"]
        FRAMES["Frame Sampler<br/>(OpenCV)<br/>Target: 1fps"]
        AUDIO_EXTRACT["Audio Extractor<br/>(moviepy)<br/>WAV output"]
    end

    subgraph Vision["👁️ Vision Analysis"]
        NSFW["NSFW Classifier<br/>(Falconsai)<br/>Per-frame inference"]
        EMOTION["Emotion Detector<br/>(DeepFace CNN)<br/>Face-aware"]
        FRAME_RESULTS["Frame Results<br/>nsfw_score, emotion<br/>confidence, quality"]
    end

    subgraph Audio["🎙️ Audio Features"]
        AUDIO_FEATURES["Feature Extraction<br/>(librosa)<br/>RMS, silence, speech-rate"]
        AUDIO_RESULTS["Audio Risk Alert<br/>volume_db, speakers<br/>silence_periods"]
    end

    subgraph Speech["🗣️ Speech-to-Text"]
        WHISPER["Whisper ASR<br/>(Primary)<br/>Long-form mode"]
        SR_FALLBACK["SpeechRecognition<br/>(Fallback)<br/>Chunked + merged"]
        TRANSCRIPT["Transcript<br/>word_data, confidence<br/>segment_meta"]
    end

    subgraph NLP["💬 NLP Safety Pipeline"]
        SEGMENT["Transcript Segmenter<br/>Sentence-level chunks<br/>with confidence"]
        PROFANITY["Profanity Check<br/>(better-profanity library)"]
        PII["PII Detection<br/>(regex patterns)<br/>phone, email, address"]
        TOXICITY["Toxicity Classifier<br/>(toxic-bert)<br/>0-1 score"]
        SENTIMENT["Sentiment Analysis<br/>(distilbert)<br/>positive/negative"]
        NER["Entity Recognition<br/>(spaCy)<br/>PERSON, ORG, etc."]
        CHAT_RESULTS["Chat/Transcript Alert<br/>reasons, confidence<br/>entities, sentiment"]
    end

    subgraph Engine["⚠️ Alert Engine"]
        SEVERITY["Severity Mapper<br/>low/medium/high/critical"]
        CONFIDENCE["Confidence Aggregator<br/>by_reason breakdown"]
        ALERT["Unified Alert<br/>run_id, source, category"]
    end

    subgraph Storage["💾 Persistence Layer"]
        MAIN_ALERTS[("alerts<br/>Primary records")]
        CHAT_TABLE[("chat_alerts<br/>Text-specific")]
        VIDEO_TABLE[("video_alerts<br/>Frame-specific")]
        AUDIO_TABLE[("audio_alerts<br/>Audio features")]
        TRANSCRIPT_TABLE[("transcript_segments<br/>Speech segments")]
        FLAGGED_FRAMES[("flagged_frames<br/>Frame evidence")]
        VALIDATION_LOGS[("validation_logs<br/>Human feedback")]
    end

    subgraph Dashboard["📊 Intelligence Layer"]
        FLASK["Flask HTML Dashboard<br/>REST API backend<br/>Custom charts"]
        EXPORT["Export Engine<br/>CSV / JSON<br/>Evaluation reports"]
    end

    VID --> FRAMES
    VID --> AUDIO_EXTRACT

    FRAMES --> NSFW
    FRAMES --> EMOTION
    NSFW --> FRAME_RESULTS
    EMOTION --> FRAME_RESULTS

    AUDIO_EXTRACT --> WHISPER
    AUDIO_EXTRACT --> SR_FALLBACK
    WHISPER --> TRANSCRIPT
    SR_FALLBACK --> TRANSCRIPT

    AUDIO_EXTRACT --> AUDIO_FEATURES
    AUDIO_FEATURES --> AUDIO_RESULTS

    TRANSCRIPT --> SEGMENT
    SEGMENT --> PROFANITY
    SEGMENT --> PII
    SEGMENT --> TOXICITY
    SEGMENT --> SENTIMENT
    SEGMENT --> NER

    PROFANITY --> CHAT_RESULTS
    PII --> CHAT_RESULTS
    TOXICITY --> CHAT_RESULTS
    SENTIMENT --> CHAT_RESULTS
    NER --> CHAT_RESULTS

    FRAME_RESULTS --> SEVERITY
    CHAT_RESULTS --> SEVERITY
    AUDIO_RESULTS --> SEVERITY

    SEVERITY --> CONFIDENCE
    CONFIDENCE --> ALERT

    ALERT --> MAIN_ALERTS
    FRAME_RESULTS --> VIDEO_TABLE
    FRAME_RESULTS --> FLAGGED_FRAMES
    CHAT_RESULTS --> CHAT_TABLE
    AUDIO_RESULTS --> AUDIO_TABLE
    TRANSCRIPT --> TRANSCRIPT_TABLE

    MAIN_ALERTS --> FLASK
    CHAT_TABLE --> FLASK
    VIDEO_TABLE --> FLASK
    AUDIO_TABLE --> FLASK
    TRANSCRIPT_TABLE --> FLASK
    FLASK --> VALIDATION_LOGS
    FLASK --> EXPORT

    style Input fill:#e1f5ff
    style Extract fill:#f3e5f5
    style Vision fill:#fff3e0
    style Audio fill:#f1f8e9
    style Speech fill:#fce4ec
    style NLP fill:#ede7f6
    style Engine fill:#fff9c4
    style Storage fill:#eceff1
    style Dashboard fill:#c8e6c9
```

### Pipeline Components Breakdown

| **Stage** | **Component** | **Input** | **Output** | **Technology** |
|---|---|---|---|---|
| Extract | Frame Sampler | Video file | Frames @ target FPS | OpenCV |
| Extract | Audio Extractor | Video file | WAV audio | moviepy |
| Vision | NSFW Classifier | Frame | {label, score} | Falconsai/nsfw_detection |
| Vision | Emotion Detector | Frame | {emotion, confidence} | DeepFace (CNN) |
| Audio | Feature Extractor | WAV | RMS, silence, speech-rate | librosa, scipy |
| Speech | Whisper ASR | WAV | Transcript + word_data | OpenAI Whisper |
| Speech | SR Fallback | WAV | Transcript + metadata | Google Speech API |
| NLP | Profanity Check | Text | {detected: bool} | better-profanity |
| NLP | PII Detection | Text | [pii_types] | regex patterns |
| NLP | Toxicity | Text | {label, score} | unitary/toxic-bert |
| NLP | Sentiment | Text | {sentiment, score} | distilbert-sst-2 |
| NLP | Entity Recognition | Text | [entities] | spaCy en_core_web_sm |
| Engine | Alert Aggregator | Analysis results | Unified alert record | Custom rule engine |
| Storage | Relational DB | Alerts + evidence | Alerts, flagged_frames, validation_logs | SQLite/PostgreSQL |
| Dashboard | Analytics Console | DB queries | Charts, filters, exports | Flask + Chart.js |

### Data Flow Layers

```
┌─────────────────────────────────────────────────────────┐
│  RAW MEDIA (Video File)                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  EXTRACTION LAYER           │
        │  ├─ Video → Frames         │
        │  └─ Video → Audio (WAV)    │
        └────────────────────────────┘
                     │
        ┌────────────┴────────────────┐
        │                             │
        ▼                             ▼
   [VISION ANALYSIS]          [AUDIO ANALYSIS]
   • NSFW Detection           • Volume/Silence
   • Emotion Detection        • Speech Rate
   • Quality Checks           • Speaker Count
        │                             │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  SPEECH-TO-TEXT            │
        │  (Whisper → SR Fallback)   │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  NLP SAFETY PIPELINE       │
        │  ├─ Profanity             │
        │  ├─ PII Detection         │
        │  ├─ Toxicity              │
        │  ├─ Sentiment             │
        │  └─ Entity Recognition    │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  ALERT ENGINE              │
        │  ├─ Severity Mapping       │
        │  ├─ Confidence Scoring     │
        │  └─ Category Assignment    │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  DATABASE PERSISTENCE      │
        │  (SQLite/PostgreSQL)       │
        │  └─ Normalized Joins       │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  DASHBOARDS & EXPORTS      │
        │  ├─ Flask HTML Dashboard   │
        │  ├─ REST API               │
        │  └─ CSV/JSON Export        │
        └────────────────────────────┘
```

---

## ✨ Features

### Core Capabilities

- ✅ **Video Frame Analysis** — Extracts and analyzes frames at configurable FPS
- ✅ **NSFW Detection** — Per-frame classification with confidence scores
- ✅ **Emotion Recognition** — Face-aware emotion detection with quality gating
- ✅ **Audio Extraction** — Robust audio extraction from video using moviepy
- ✅ **Speech-to-Text** — Bilingual support with Whisper + SpeechRecognition fallback
- ✅ **Transcript Safety** — Full NLP pipeline applied to transcribed text
- ✅ **Normalized Alerts** — Unified alert schema with severity and confidence
- ✅ **Rule-Based Flagging** — Configurable thresholds, pattern detection, heuristics

### Advanced Capabilities

- ✅ **Temporal Stabilization** — EMA smoothing for NSFW decisions to reduce false positives
- ✅ **Batch Inference** — Optimized frame and text batch processing for throughput
- ✅ **Quality-Aware Emotion** — Skips blurry/dark frames; optional face-ROI-first detection
- ✅ **Transcript Segmentation** — Sentence-level chunking with confidence propagation
- ✅ **Multi-Analyzer Fusion** — Combines vision, audio, and text risk signals
- ✅ **Run-Level Lineage** — Tracks all alerts with `run_id` for reproducibility
- ✅ **Confidence-by-Reason** — Maps each reason type to its contributing model score

### Dashboard & Visibility

- ✅ **Flask HTML Dashboard** — Custom-styled UI with REST API, real-time refresh, responsive design
- ✅ **Upload & Review Page** — Live upload, progress, video preview, transcript review, and validation
- ✅ **Evaluation Reporting** — Confusion matrices, accuracy/precision/recall/F1 metrics, profiling

---

## Upload & Review Workflow (HTML Dashboard)

1. Start the HTML dashboard:

```bash
python html_dashboard.py
```

2. Open the Upload & Review page:

```
http://localhost:8502/dashboard/upload
```

3. Upload a video and monitor:
    - Live progress via `/status/<run_id>`
    - Video preview via `/video/<run_id>`
    - Flagged frames + validation controls
    - Transcript Review panel with search and flagged-only filter

### Key Paths

- `uploads/` — Uploaded video files
- `static/frames/` — Saved flagged-frame thumbnails

---

## HTML Dashboard API (Selected)

- `POST /upload` — Upload a video, returns `run_id`
- `GET /status/<run_id>` — Run progress + current frame info
- `GET /video/<run_id>` — Stream uploaded video for preview
- `GET /api/flagged-items` — Flagged frames + audio/text events
- `GET /api/transcript` — Transcript segments for the run
- `POST /validate` — Human feedback (correct/incorrect) for a flagged frame
- `GET /api/dashboard-data` — Charts + KPI data

### Scalability & Deployment

- ✅ **SQLite Default** — Lightweight, file-based persistence for single-machine deployments
- ✅ **PostgreSQL Optional** — Drop-in alternative for multi-instance/cloud deployments
- ✅ **Environment Configuration** — 20+ feature flags and environment variables for tuning
- ✅ **Containerization-Ready** — Clean dependency isolation, virtual environment support

---

## 🆕 Latest Improvements

### 1️⃣ Model Performance Enhancements

- **Batch NSFW Inference** — Reduced per-frame model call overhead; improved FPS throughput
- **Temporal NSFW Smoothing** — EMA-based decision stabilization with configurable alpha
- **High-Confidence Bypass** — Critical NSFW frames bypass temporal requirement for responsiveness
- **Segment-Level Confidence** — Transcript NLP checks propagate word-level confidence scores

### 2️⃣ Preprocessing & Feature Extraction

**Frame-Level:**
- Adaptive resize for GPU memory efficiency
- Optional CLAHE contrast normalization for challenging lighting
- Blur variance gating to skip noisy/low-quality frames
- Brightness checks to avoid inference on extremely dark frames

**Audio-Level:**
- RMS-based volume anomaly detection
- Silence period tracking with configurable thresholds
- Speech rate estimation via onset detection (syllabic proxy)
- Speaker change detection via spectral centroid analysis

**Quality Metrics:**
- Per-frame blur variance and brightness tracking
- Online quality statistics for adaptive thresholds
- Emotion-to-sentiment mapping with context awareness

### 3️⃣ Pipeline Integration Improvements

- **Shared Audio Payload** — Single audio extraction reused across video, transcript, and audio analyzers
- **Transcription Fallback Chain** — Whisper (primary) → chunked SpeechRecognition with overlap merging
- **Run-ID Propagation** — All alerts tagged with `run_id` for multi-run analytics and comparison
- **Unified Confidence Mapping** — Per-reason confidence scores in `confidence_by_reason` dict

### 4️⃣ AI/ML Model Stack

**NLP Models:**
- `unitary/toxic-bert` — Binary toxicity classification with 0-1 confidence
- `distilbert-base-uncased-finetuned-sst-2-english` — Sentiment (positive/negative)
- `openai/whisper-base` (configurable size) — Multilingual ASR, long-form audio
- `spacy/en_core_web_sm` — Named entity recognition (PERSON, ORG, LOCATION, etc.)

**Vision Models:**
- `Falconsai/nsfw_image_detection` — Binary NSFW classification per frame
- `DeepFace` — CNN-based facial emotion inference (7 classes)
- `OpenCV Haar Cascade` — Face detection for face-ROI-aware emotion analysis

**Audio Processing:**
- **librosa** — MFCC, RMS, onset detection, spectral features
- **scipy** — Signal processing, spectral centroid
- **pydub / moviepy** — Audio extraction and format conversion

### 5️⃣ Scalability & Optimization

- **Batched Model Inference** — Frame batches processed by NSFW model in single forward pass
- **Transcript Chunking** — Long transcripts split by sentence with optional hard-wrap fallback
- **Database Write Optimization** — Batched inserts with single transaction commit
- **Optional PostgreSQL Backend** — Via environment variable for multi-instance deployments
- **Dashboard Caching** — 2-second TTL on alert queries to reduce DB load

### 6️⃣ Upload & Review Experience

- **Dedicated Upload Module** — Upload routes isolated in `upload_module.py`
- **Live Progress Polling** — `status` API powers real-time progress bar
- **Video Preview While Processing** — Upload page streams the current run
- **Flagged Frame Evidence** — Persisted in `flagged_frames` with thumbnails
- **Human Validation Logs** — Stored in `validation_logs` for QA
- **Transcript Review Panel** — Search, filter, and inspect transcript segments

---

## 🧰 Tech Stack & Dependencies

| **Layer** | **Component** | **Technology** |
|---|---|---|
| **Frontend** | Web UI | HTML5, CSS3, JavaScript (ES6+), Chart.js |
| **Frontend** | Responsive Design | CSS Grid, Flexbox, media queries |
| **Dashboard** | HTML Analytics | Flask 3.0+, Chart.js 4.4+ |
| **Dashboard** | Custom UI | Flask 3.0+, Jinja2 templating |
| **Backend** | REST API | Flask 3.0+ with JSON endpoints |
| **Backend** | Concurrency | Python threading, asyncio |
| **Backend** | Configuration | python-dotenv, environment variables |
| **AI/ML** | NLP | Hugging Face Transformers 4.35+, spaCy 3.7+ |
| **AI/ML** | NLP Utils | better-profanity 0.7+, regex |
| **AI/ML** | Vision | OpenCV 4.8+, DeepFace 0.0.79+, Falconsai classifier |
| **AI/ML** | Audio | librosa 0.10+, scipy 1.11+, pydub 0.25+ |
| **AI/ML** | ASR | OpenAI Whisper (HF Transformers), SpeechRecognition 3.10+ |
| **AI/ML** | Inference | PyTorch 2.0+, TensorFlow/tf-keras 2.15+ |
| **Database** | Primary | SQLite 3 (file-based) |
| **Database** | Scale-out | PostgreSQL 12+ (optional via environment) |
| **Database** | Driver | psycopg2-binary 2.9+ |
| **Data** | Processing | pandas 2.0+, NumPy 1.24+ |
| **Data** | Visualization | Plotly 5.17+, Chart.js 4.4+ |
| **Testing** | Framework | unittest, unittest.mock |
| **Build** | Package Manager | pip, setuptools |
| **Build** | Virtual ENV | venv, virtualenv |
| **DevOps** | Runtime | Python 3.10+, ffmpeg 5.0+ |
| **IDE** | Recommended | VS Code with Python extension |
| **SCM** | Version Control | Git |

---

## ⚙️ Installation Guide (VS Code Friendly)

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Git** for version control
- **ffmpeg 5.0+** installed on system PATH

#### Install ffmpeg:

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
```

### Step-by-Step Setup

```bash
# 1) Clone repository
git clone <your-repo-url>
cd melodywings_guard

# 2) Create virtual environment
python -m venv .venv

# 3) Activate virtual environment
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux bash
source .venv/bin/activate

# 4) Install Python dependencies
pip install -r requirements.txt

# 5) Download spaCy model
python -m spacy download en_core_web_sm

# 6) Verify installation
python -c "import cv2, transformers, spacy; print('✅ All imports successful')"
```

---

## ▶️ Usage Guide

### 1) Run Full Pipeline

```bash
python main.py
```

**Pipeline stages executed:**
1. Chat moderation on sample test messages
2. Video frame analysis (NSFW + emotion detection)
3. Audio extraction and speech transcription
4. Transcript NLP safety analysis (toxicity, PII, etc.)
5. Audio feature anomaly analysis
6. Alert persistence to SQLite database
7. Summary metrics and confusion matrix export

**Output artifacts:**
- `melodywings_guard.db` — SQLite database with all alerts
- `melodywings_guard.log` — Detailed execution log
- `chat_confusion_matrix.html` — Interactive evaluation visualization

### 2) Launch Flask HTML Dashboard

```bash
# Start Flask server
python html_dashboard.py

# Opens at http://localhost:8502
```

**Features:**
- Beautiful, responsive custom HTML UI
- REST API backend for flexibility
- Real-time KPI cards and charts (Chart.js)
- Page-based navigation (Overview, Chat, Video, Audio)
- Alert drilldown panel with detailed inspection
- CSV/JSON export with filtering
- Configurable auto-refresh interval

### 3) Run Unit Tests

```bash
# Test chat analyzer
python test_chat_analyzer.py

# Test video analyzer
python test_video_analyzer.py
```

**Test coverage:**
- Profanity, PII, toxicity, sentiment detection
- Frame chunking, transcript segmentation
- Evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix generation

### Configuration & Customization

```bash
# Run with clear-on-startup behavior
MWG_CLEAR_ON_RUN=true python main.py

# Override Whisper model
MWG_WHISPER_MODEL=openai/whisper-small python main.py

# Use PostgreSQL instead of SQLite
DB_BACKEND=postgres DB_POSTGRES_DSN="postgresql://user:pass@host/db" python main.py

# Enable strict mode (fail on any model load error)
STRICT=true python main.py

# Protect dashboard/API endpoints with token auth
MWG_REQUIRE_API_AUTH=true MWG_API_AUTH_TOKEN=change-me python html_dashboard.py

# Set max upload size (MB) for /upload endpoint
MWG_MAX_UPLOAD_MB=512 python html_dashboard.py
```

### Security Controls (Dashboard/API)

- `MWG_REQUIRE_API_AUTH` — Set to `true` to require API auth on upload/API/status/video routes.
- `MWG_API_AUTH_TOKEN` — Shared bearer token (also enables auth if set).
- `MWG_API_AUTH_QUERY_PARAM` — Query parameter used to bootstrap browser auth cookie (default: `api_key`).
- `MWG_API_AUTH_COOKIE_NAME` — Cookie name for persisted browser auth token (default: `mwg_api_token`).
- `MWG_MAX_UPLOAD_MB` — Maximum accepted upload size in MB (default: `512`).

When auth is enabled, open dashboard URLs with the token once to establish the cookie:

```text
http://localhost:8502/dashboard/upload?api_key=change-me
```

---

## 📁 Project Structure

```
melodywings_guard/
├── 📄 main.py                          # Pipeline orchestrator entry point
├── 📄 chat_analyzer.py                 # NLP safety analysis (profanity, PII, toxicity, sentiment, NER)
├── 📄 video_analyzer.py                # Video frame & transcript analysis pipeline
├── 📄 audio_analyzer.py                # Audio feature extraction and anomaly detection
├── 📄 alert_engine.py                  # Unified alert logger and severity mapping
├── 📄 database.py                      # SQLite/PostgreSQL adapter and schema
├── 📄 html_dashboard.py                # Flask API + custom HTML dashboard
├── 📄 requirements.txt                 # Python dependencies
├── 📄 test_chat_analyzer.py            # Unit tests for chat analyzer
├── 📄 test_video_analyzer.py           # Unit tests for video analyzer
├── 📄 README.md                        # This file
├── 📄 LICENSE                          # MIT License
│
├── 📁 static/                          # Frontend assets
│   ├── dashboard.css                   # HTML dashboard styling
│   └── dashboard.js                    # HTML dashboard frontend logic
│
├── 📁 templates/                       # Jinja2 templates
│   └── dashboard.html                  # Flask dashboard template
│
├── 💾 melodywings_guard.db             # SQLite database (generated at runtime)
├── 📋 melodywings_guard.log            # Execution log (generated at runtime)
├── 📊 chat_confusion_matrix.html       # Evaluation visualization (generated at runtime)
└── 📄 alerts_log.json                  # JSON alert archive (generated at runtime)
```

---

## 📊 Performance Metrics

### Measured Results (Latest Runs)

| **Metric** | **Value** | **Source** |
|---|---:|---|
| **Chat Accuracy** | 1.00 (100%) | Confusion matrix (TN=5, FP=0, FN=0, TP=5) |
| **Chat Precision** | 1.00 | Same evaluation set |
| **Chat Recall** | 1.00 | Same evaluation set |
| **Chat F1-Score** | 1.00 | Same evaluation set |
| **Video Effective FPS** | 4.56–7.55 (avg 6.3) | 36 historical video runs |
| **Per-Frame Latency** | 132–219 ms | Derived from effective FPS |
| **Frame Flagging Rate** | 0.00%–70.31% (tuned) | Before/after optimization |

### Performance Trends

**Chat Analysis Pipeline:**
- Tokenization + batched model inference: **~50–100 ms per message**
- Full NLP pipeline (5 stages): **~200–500 ms per message**
- Batch processing 10 messages: ~1.5–2.0 seconds

**Video Analysis Pipeline:**
- Frame extraction + resize: **~10–30 ms per frame**
- NSFW inference (batched): **~20–40 ms per batch of 8 frames**
- Emotion inference: **~40–100 ms per frame** (face detection + CNN)
- Full pipeline (256 frames): **~30–50 seconds**

**Database Operations:**
- Single alert insert: **~2–5 ms**
- Batched 100 inserts with commit: **~20–30 ms**
- Full join query (all alerts): **~50–200 ms**

### Historical Improvements

| **Iteration** | **Flagging Rate** | **FPS** | **Notes** |
|---|---:|---:|---|
| Early tuning | 70.31% | 6.0–7.0 | High false positives on NSFW |
| Mid-tuning | 10–20% | 6.0–7.5 | Temporal smoothing added |
| Latest (tuned) | 0.00%–0.39% | 4.56–7.55 | EMA + consecutive confirmation |

---

## 🚀 Future Enhancements

### Near-Term (Next Sprint)

- [ ] **Real-Time Streaming** — WebRTC/RTSP/RTMP ingestion support
- [ ] **GPU Acceleration** — CUDA/cuDNN optimization for vision models
- [ ] **Multi-File Upload** — Batch video processing with job queue
- [ ] **Persistent Config** — UI-based threshold and model selection tuning

### Medium-Term

- [ ] **Advanced Models** — Vision Transformers, LSTM sequence analysis, multimodal fusion
- [ ] **Explainability** — SHAP/LIME-based reason justification for flagged items
- [ ] **Analyst Workflows** — Case management, saved views, cross-alert correlation
- [ ] **Containerization** — Docker image, docker-compose orchestration
- [ ] **Cloud Deployment** — Azure Container Apps, AWS ECS, GCP Cloud Run templates

### Long-Term

- [ ] **Federated Learning** — Collaborative model improvement across deployments
- [ ] **Custom Fine-Tuning** — Transfer learning on user-uploaded ground-truth data
- [ ] **Automated Remediation** — Content redaction, watermarking, blur pipelines
- [ ] **Compliance Reporting** — GDPR, CCPA, content moderation audit trails

---

## 🤝 Contributing

Contributions are **welcome and encouraged**!

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "Add: descriptive feature message"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** with:
   - Clear problem statement
   - Implementation summary
   - Test evidence (unit tests, screenshots, metrics)
   - Link to any related issues

### Contribution Guidelines

- ✅ **Keep changes modular** — One feature per PR
- ✅ **Write tests** — All logic changes require unit test coverage
- ✅ **Update documentation** — README, docstrings, comments
- ✅ **Preserve backwards compatibility** — Don't break existing APIs without migration path
- ✅ **Follow existing style** — Match code conventions in the file
- ✅ **Performance-conscious** — Profile changes; avoid regressions

### Areas for Contribution

- 🐛 **Bug fixes** — Found a crash or incorrect behavior?
- 🎨 **UI/UX improvements** — Dashboard enhancements, new charts
- 🚀 **Performance optimization** — Faster inference, better batch processing
- 📚 **Documentation** — Better guides, tutorials, examples
- 🧪 **Test coverage** — Additional unit/integration tests
- 🌍 **Localization** — Multi-language support

---

## 📜 License

This project is licensed under the **MIT License** — permitting free use, modification, and distribution in both commercial and non-commercial settings.

See [LICENSE](LICENSE) file for full details.

**Recommended for:**
- ✅ Educational projects
- ✅ Hackathon submissions
- ✅ Portfolio demonstration
- ✅ Research and experimentation
- ✅ Commercial deployment (with attribution)

---

## 🙌 Acknowledgements

This project stands on the shoulders of exceptional open-source ecosystems:

- **[Hugging Face Transformers](https://huggingface.co/transformers/)** — State-of-the-art NLP models
- **[OpenCV](https://opencv.org/)** — Computer vision foundation
- **[DeepFace](https://github.com/serengp/deepface)** — Facial recognition and emotion analysis
- **[spaCy](https://spacy.io/)** — Industrial-strength NLP
- **[Flask](https://flask.palletsprojects.com/)** — Lightweight web framework
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Robust speech-to-text
- **[librosa](https://librosa.org/)** — Audio analysis and music information retrieval

---

## 📬 Support & Questions

- 💬 **GitHub Issues** — Report bugs and request features
- 🔗 **GitHub Discussions** — Ask questions and share ideas
- 📧 **Contribution** — Send pull requests

---

<p align="center">
  <b>Made with ❤️ for content safety research, education, and production deployment</b>
  <br/><br/>
  <a href="#-project-overview">⬆️ Back to top</a>
</p>
