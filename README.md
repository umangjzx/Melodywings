# MelodyWings Guard

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/AI%2FML-Transformers%20%7C%20DeepFace-FF6F61" alt="AI/ML" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Backend-Flask%20%7C%20Streamlit-0E1117" alt="Backend" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Database-SQLite%20%7C%20PostgreSQL-336791" alt="Database" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Production%20Ready%20Prototype-2EA44F" alt="Status" /></a>
</p>

Multimodal content-safety platform that analyzes **video, audio, and text** end-to-end.  
It turns raw media into structured moderation intelligence using a pipeline of computer vision, speech recognition, NLP, alert scoring, and dashboard analytics.

---

## 🎯 Project Overview

MelodyWings Guard processes uploaded video content and automatically identifies potential safety risks through a synchronized multimodal pipeline:

- **Visual analysis** of sampled frames (NSFW + emotion signals)
- **Audio behavior analysis** (volume, silence, speech rate, speaker-change heuristics)
- **Speech-to-text transcription** from extracted audio
- **NLP moderation** on transcript text (toxicity, profanity, PII, sentiment, entity checks)
- **Unified alerting + storage** with severity, confidence, and category metadata

---

## 🏗️ System Architecture (Detailed)

### End-to-End Data Flow

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
        NSFW["NSFW Classifier<br/(Falconsai model)<br/>Per-frame inference"]
        EMOTION["Emotion Detector<br/>(DeepFace CNN)<br/>Face-aware"]
        FRAME_RESULTS["Frame Results<br/>nsfw_score, emotion<br/>confidence, quality"]
    end

    subgraph Audio["🎙️ Audio Analysis"]
        AUDIO_FEATURES["Feature Extraction<br/>(librosa)<br/>RMS, silence, speech-rate"]
        AUDIO_RESULTS["Audio Alert<br/>volume_db, speakers<br/>silence_periods"]
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
        MAIN_ALERTS[("alerts table<br/>Primary records")]
        CHAT_TABLE[("chat_alerts<br/>Text-specific")]
        VIDEO_TABLE[("video_alerts<br/>Frame-specific")]
        AUDIO_TABLE[("audio_alerts<br/>Audio features")]
        TRANSCRIPT_TABLE[("transcript_segments<br/>Speech segments")]
    end

    subgraph Dashboard["📊 Intelligence Layer"]
        STREAMLIT["Streamlit Dashboard<br/>Advanced filters<br/>Analytics + export"]
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
    CHAT_RESULTS --> CHAT_TABLE
    AUDIO_RESULTS --> AUDIO_TABLE
    TRANSCRIPT --> TRANSCRIPT_TABLE

    MAIN_ALERTS --> STREAMLIT
    CHAT_TABLE --> STREAMLIT
    VIDEO_TABLE --> STREAMLIT
    AUDIO_TABLE --> STREAMLIT
    TRANSCRIPT_TABLE --> STREAMLIT

    MAIN_ALERTS --> FLASK
    CHAT_TABLE --> FLASK
    VIDEO_TABLE --> FLASK
    AUDIO_TABLE --> FLASK
    TRANSCRIPT_TABLE --> FLASK

    STREAMLIT --> EXPORT
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

| Stage | Component | Input | Output | Model/Tech |
|---|---|---|---|---|
| **Extract** | Frame Sampler | Video | Frames @ target FPS | OpenCV |
| **Extract** | Audio Extractor | Video | WAV | moviepy |
| **Vision** | NSFW Classifier | Frame | {label, score} | Falconsai |
| **Vision** | Emotion Detector | Frame | {emotion, confidence} | DeepFace |
| **Audio** | Feature Extractor | WAV | RMS, silence, speech-rate | librosa |
| **Speech** | Whisper ASR | WAV | Transcript + confidence | OpenAI |
| **Speech** | SR Fallback | WAV | Transcript + word data | Google Speech API |
| **NLP** | Profanity | Text | {detected: bool} | better-profanity |
| **NLP** | PII | Text | [pii_types] | regex |
| **NLP** | Toxicity | Text | {label, score} | toxic-bert |
| **NLP** | Sentiment | Text | {sentiment, score} | distilbert |
| **NLP** | Entity Recognition | Text | [entities] | spaCy |
| **Engine** | Alert Aggregator | Analysis results | Unified alert | Rule engine |
| **Storage** | Relational DB | Alerts | Queryable records | SQLite/PostgreSQL |
| **Dashboard** | Analytics Console | DB | Charts, filters, export | Streamlit + Flask |

### Key Data Transformations

```
Video (raw)  
  ↓  
{frames: [...], audio_path: str}  
  ↓  
{nsfw_scores: [...], emotions: [...], transcript: str, audio_features: {...}}  
  ↓  
{chat_alerts: [...], video_alerts: [...], audio_alerts: [...], transcript_results: [...]}  
  ↓  
 Alert records (SQLite: alerts + detail tables)  
  ↓  
{Streamlit dashboard, Flask API, CSV/JSON exports}
```

---

## ✨ Features

### 1) Model Performance Improvements

- Added **batch NSFW inference** to reduce model call overhead and improve throughput
- Introduced **temporal NSFW smoothing (EMA)** + **consecutive confirmation logic** to reduce noisy spikes
- Added **high-confidence bypass** for critical NSFW frames to preserve sensitivity on strong signals
- Improved transcript safety classification with **segment-level confidence-aware NLP checks**

### 2) Enhanced Preprocessing & Feature Extraction

- Added frame preprocessing pipeline:
  - adaptive resize for faster inference
  - optional CLAHE contrast normalization
- Added frame quality gating for emotion analysis:
  - blur variance checks
  - brightness checks
  - optional face-region-first emotion inference
- Upgraded audio analysis heuristics for:
  - loudness anomalies
  - silence pattern detection
  - speech-rate agitation signals
  - speaker-change estimation

### 3) Improved Pipeline Integration

- Introduced **shared audio extraction payload** reused across video, transcript, and audio analyzers
- Added robust transcription fallback chain:
  - Whisper (primary)
  - chunked SpeechRecognition fallback with overlap merging
- Added run-level traceability with **`run_id`** propagation across all alert tables

### 4) New / Upgraded AI-ML Models

- **Transformers**:
  - `unitary/toxic-bert` for toxicity
  - `distilbert-base-uncased-finetuned-sst-2-english` for sentiment
  - `openai/whisper-base` (configurable) for speech-to-text
- **Vision models**:
  - `Falconsai/nsfw_image_detection` for frame safety
  - `DeepFace` emotion inference (CNN-based facial feature extraction)
- **NLP enrichment**:
  - spaCy `en_core_web_sm` for entity extraction

### 5) Scalability & Optimization Upgrades

- Batch processing for frame inference and chat NLP operations
- Transcript chunking for long-text robustness and memory-safe processing
- Database write optimization with batched commits
- Optional PostgreSQL backend support for scale-out persistence
- Short-lived dashboard caching to improve API response performance

---

### Core Features

- ✅ Video frame extraction and analysis
- ✅ NSFW detection with confidence scoring
- ✅ Emotion-to-sentiment mapping for visual behavior
- ✅ Audio extraction and speech transcription
- ✅ Transcript moderation using NLP safety pipeline
- ✅ Alert engine with severity and category classification
- ✅ Local persistence with relational schema and drill-down joins

### Advanced Features

- ✅ Temporal stabilization for NSFW decisions (EMA smoothing)
- ✅ Face-aware, quality-aware emotion detection
- ✅ Long transcript segmentation with confidence metadata
- ✅ Batch processing for vision and NLP inference
- ✅ Robust transcription fallback chain (Whisper → SpeechRecognition)
- ✅ Evaluation utilities:
  - confusion matrix generation
  - accuracy / precision / recall / F1 metrics
- ✅ Dual dashboards:
  - Streamlit analytics console with advanced filtering
  - Flask + custom HTML operations dashboard
  - REST API for programmatic access
- ✅ CSV/JSON export and rich filtering for investigation workflows
- ✅ Multi-run tracking with run_id lineage
- ✅ PostgreSQL backend support for scale-out deployment

---

## 🆕 Latest Improvements

---

## 🧰 Tech Stack & Dependencies

| Layer | Component | Technologies |
|---|---|---|
| **Frontend** | Web UI | HTML5, CSS3, JavaScript, Chart.js |
| **Dashboard** | Streaming | Streamlit + Plotly |
| **Dashboard** | Custom HTML | Flask + Jinja2 templates |
| **Backend** | API Server | Flask (REST endpoints) |
| **Backend** | Orchestration | Python asyncio, threading |
| **AI/ML** | NLP Safety | Hugging Face Transformers, spaCy, better-profanity |
| **AI/ML** | Vision Safety | OpenCV, DeepFace (face detection), NSFW classifier |
| **AI/ML** | Speech-to-Text | OpenAI Whisper, Google Speech Recognition |
| **AI/ML** | Audio Features | librosa, scipy, pydub, numpy |
| **Database** | Default | SQLite (local, file-based) |
| **Database** | Scale-out | PostgreSQL (optional, via environment) |
| **Data** | Processing | pandas, NumPy |
| **Data** | Visualization | Plotly, Chart.js |
| **Testing** | Framework | unittest, mock |
| **DevOps** | Runtime | Python 3.10+, ffmpeg, pip, venv |
| **IDE** | Recommended | VS Code with Python extension |

---

## 🏗️ System Architecture

### Pipeline Flow

**Video Input → Frame Extraction → Audio Extraction → Speech-to-Text → NLP → Insights**

```mermaid
flowchart LR
    A[Video Input] --> B[Frame Extraction]
    A --> C[Audio Extraction]
    C --> D[Speech-to-Text]
    D --> E[NLP Safety Analysis]
    B --> F[Vision Risk Analysis\nNSFW + Emotion]
    E --> G[Unified Alert Engine]
    F --> G
    C --> H[Audio Feature Analysis]
    H --> G
    G --> I[(SQLite / PostgreSQL)]
    I --> J[Insights Dashboard\nFlask + Streamlit]
```

---

## ⚙️ Installation Guide (VS Code Friendly)

### Prerequisites

- Python **3.10+**
- Git
- ffmpeg installed on system PATH

Install ffmpeg:

- Windows: `winget install ffmpeg`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

### Setup Steps

```bash
# 1) Clone repository
git clone <your-repo-url>
cd melodywings_guard

# 2) Create virtual environment
python -m venv .venv

# 3) Activate virtual environment
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 4) Install dependencies
pip install -r requirements.txt

# 5) Install spaCy model
python -m spacy download en_core_web_sm
```

---

## ▶️ Usage

### 1) Run Full Pipeline

```bash
python main.py
```

What this run performs:

1. Chat moderation on sample messages
2. Video frame analysis (NSFW + emotion)
3. Audio extraction and speech transcription
4. Transcript NLP safety analysis
5. Audio feature anomaly analysis
6. Alert persistence + summary metrics
7. Confusion matrix export for chat evaluation

### 2) Launch HTML Dashboard

```bash
python html_dashboard.py
```

Open: **http://localhost:8502**

### 3) Launch Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### 4) Run Tests

```bash
python test_chat_analyzer.py
python test_video_analyzer.py
```

### Notes

- Update `VIDEO_PATH` in `main.py` to your local video file before running full video analysis.
- Output artifacts include:
  - `melodywings_guard.db`
  - `melodywings_guard.log`
  - `chat_confusion_matrix.html`

---

## 📁 Project Structure

```text
melodywings_guard/
├── alert_engine.py              # Unified alert logger and severity mapping
├── audio_analyzer.py            # Audio feature extraction and anomaly rules
├── chat_analyzer.py             # NLP moderation (toxicity, PII, sentiment, entities)
├── dashboard.py                 # Streamlit analytics dashboard
├── database.py                  # SQLite/PostgreSQL adapter and schema layer
├── html_dashboard.py            # Flask API + custom HTML dashboard server
├── main.py                      # Pipeline orchestrator entry point
├── video_analyzer.py            # Frame/video/transcript analysis pipeline
├── requirements.txt             # Python dependencies
├── test_chat_analyzer.py        # Unit tests for chat analyzer
├── test_video_analyzer.py       # Unit tests for video analyzer
├── alerts_log.json              # Alert export/log artifact
├── analysis_output.txt          # Pipeline run output artifact
├── chat_confusion_matrix.html   # Chat evaluator visualization artifact
├── static/
│   ├── dashboard.css            # HTML dashboard styling
│   └── dashboard.js             # HTML dashboard frontend logic
└── templates/
    └── dashboard.html           # HTML dashboard template
```

---

## 📊 Performance Metrics

### Current Measured Results

| Metric | Current Value | Source |
|---|---:|---|
| Chat Accuracy | **1.00** | `chat_confusion_matrix.html` (TN=5, FP=0, FN=0, TP=5) |
| Chat Precision | **1.00** | Same evaluation set |
| Chat Recall | **1.00** | Same evaluation set |
| Chat F1-score | **1.00** | Same evaluation set |
| Video Effective FPS | **4.56 to 7.55** (avg **6.3**) | Historical run logs |
| Approx. Per-frame Latency | **132 ms to 219 ms** | Derived from effective FPS |

### Improvements Over Previous Iterations

- Frame false-positive behavior improved significantly in logged runs:
  - earlier worst-case observed: **180/256 flagged (70.31%)**
  - recent tuned runs: **0 to 1/256 flagged (0.00% to 0.39%)**
- Throughput stabilized with batched frame inference and optimized preprocessing
- End-to-end pipeline now reports structured runtime telemetry for continuous tuning

### Metrics Tracked Per Run

- Accuracy / Precision / Recall / F1 (chat, optional video GT)
- Frame stage runtime and effective FPS
- Average frame processing latency
- NSFW and emotion inference timing
- Transcript stage runtime and flagged-segment count

---

## 🚀 Future Enhancements

- Real-time stream ingestion (WebRTC/RTSP/WebSocket pipelines)
- Stronger multimodal models (action/context-aware risk detection)
- Optional LSTM or temporal transformers for sequence-level behavior modeling
- Better explainability for why an item was flagged
- Enhanced UI with analyst workflows, saved views, and triage queues
- Containerized deployment (Docker + CI/CD)
- Cloud deployment profiles (Azure/AWS/GCP)

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Open a Pull Request with:
   - problem statement
   - implementation summary
   - test evidence/screenshots

### Contribution Guidelines

- Keep changes modular and well-documented
- Add or update tests for logic changes
- Preserve backwards compatibility where possible
- Follow existing code style and naming conventions

---

## 📜 License

This project is intended for educational, hackathon, and portfolio use.

Recommended open-source license: **MIT License**.

---

## 🙌 Acknowledgements

- Hugging Face Transformers ecosystem
- OpenCV and DeepFace communities
- spaCy NLP ecosystem
- Streamlit and Flask maintainers
