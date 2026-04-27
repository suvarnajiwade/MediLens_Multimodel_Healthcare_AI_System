# MediLens — Multimodal Healthcare AI System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LangChain-0.2-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/LangGraph-0.1-red?style=flat-square" />
  <img src="https://img.shields.io/badge/vLLM-0.4-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  An end-to-end AI platform that analyzes medical reports, diagnoses skin conditions, recommends specialists, and books appointments — powered by LangGraph, LangChain, vLLM, and computer vision.
</p>

---

## What is MediLens?

MediLens is a production-grade multimodal healthcare AI system that helps patients understand their medical reports, get AI-powered skin condition analysis, find nearby specialists, and book appointments — all in one platform.

Patients upload a lab report PDF, a scanned document, or a skin photo. MediLens extracts clinical entities, flags abnormalities, generates a plain-language explanation, scores urgency, and routes the patient to the right specialist automatically.

---

## Key Features

- **Medical Report Analysis** — Parses PDF and scanned reports using OCR with automatic vLLM vision fallback for low-quality scans
- **Skin Condition Diagnosis** — EfficientNet-B4 fine-tuned on HAM10000 + ISIC 2020 classifies 9 skin conditions with severity scoring
- **vLLM Multimodal Description** — LLaVA / Qwen-VL generates clinical descriptions from images when OCR confidence is low
- **Medical NER** — BioBERT + scispaCy extracts biomarkers, diagnoses, medications, and lab values
- **Abnormality Detection** — Flags out-of-range values against LOINC reference ranges with urgency scoring (1–10)
- **RAG-Grounded Answers** — LangChain RAG pipeline over PubMed, SNOMED-CT, DermNet for accurate, grounded responses
- **Nearby Doctor Finder** — Google Maps Places API finds specialists by location, rating, and availability
- **AI Specialist Recommendation** — LLM maps report findings to the correct medical specialty
- **Smart Appointment Booking** — Urgency-based slot selection with SMS + email reminders via Twilio and SendGrid
- **Multilingual Support** — Explains reports in English, Hindi, Marathi, Tamil, Bengali, and Telugu
- **Doctor Review Portal** — Human-in-the-loop validation where doctors override or approve AI analysis
- **Biomarker Trend Tracking** — Historical report comparison with ML-based trend forecasting
- **LangGraph Orchestration** — Full pipeline managed as a stateful graph with conditional routing
- **MLflow Experiment Tracking** — Model versioning, OCR accuracy, NER F1, and CV metrics tracked end-to-end

---

## System Architecture

```
User Input (PDF / Scan / Skin Photo / Voice / Symptoms)
                        │
                        ▼
          ┌─────────────────────────┐
          │   LangGraph Router      │
          │   (State Machine)       │
          └────────┬────────────────┘
          │        │        │
     Document   Skin Img  Query
          │        │        │
          ▼        ▼        ▼
      OCR + NER  EfficientNet  Whisper STT
      BioBERT    + vLLM Vision  + Intent Classifier
      vLLM fallback  Severity Score
          │        │        │
          └────────┴────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │  LangChain RAG Layer  │
       │  Pinecone · PubMed   │
       │  SNOMED · DermNet    │
       └───────────┬───────────┘
                   │
                   ▼
       LLM Explainer + Urgency Scorer
                   │
                   ▼
       Specialist Recommender
                   │
                   ▼
       Doctor Finder + Appointment Engine
       (Google Maps API + Celery + Twilio)
                   │
                   ▼
       Patient Dashboard + Doctor Portal
       (React + FastAPI + PostgreSQL)
                   │
                   ▼
       MLOps Layer (MLflow · DVC · Docker · CI/CD)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS, React Query |
| API | FastAPI (async), Pydantic v2 |
| Orchestration | LangGraph (stateful pipeline graph) |
| LLM Framework | LangChain (RAG, chains, memory, tools) |
| LLM Inference | vLLM — LLaVA-1.6, Llama-3, Qwen-VL |
| OCR | Tesseract, PaddleOCR |
| PDF Parsing | PyMuPDF (fitz) |
| Medical NER | scispaCy, BioBERT (HuggingFace) |
| Skin CV Model | EfficientNet-B4 (PyTorch + timm) |
| Embeddings | BioBERT / text-embedding-ada-002 |
| Vector Store | Pinecone (prod), ChromaDB (dev) |
| Database | PostgreSQL + pgvector |
| Task Queue | Celery + Redis |
| Speech-to-Text | OpenAI Whisper |
| Maps | Google Maps Places API |
| Notifications | Twilio (SMS), SendGrid (email) |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | AWS EKS / GCP GKE |
| Monitoring | Prometheus + Grafana |

---

## Project Structure

```
medilens/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── report.py        # Report upload & analysis endpoints
│   │   │   ├── skin.py          # Skin image analysis endpoints
│   │   │   ├── doctor.py        # Doctor finder endpoints
│   │   │   └── appointment.py   # Appointment booking endpoints
│   ├── graph/
│   │   ├── state.py             # LangGraph MedicalState definition
│   │   ├── router.py            # Input type router node
│   │   ├── ocr_node.py          # OCR + vLLM fallback node
│   │   ├── ner_node.py          # Medical NER node
│   │   ├── skin_node.py         # Skin CV + vLLM describe node
│   │   ├── query_node.py        # Query intent classifier node
│   │   ├── rag_node.py          # LangChain RAG node
│   │   ├── llm_node.py          # LLM explainer + urgency node
│   │   ├── doctor_node.py       # Specialist recommender node
│   │   └── appointment_node.py  # Appointment engine node
│   ├── models/
│   │   ├── skin_classifier.py   # EfficientNet-B4 wrapper
│   │   ├── ner_model.py         # BioBERT / scispaCy wrapper
│   │   └── vllm_client.py       # vLLM API client
│   ├── rag/
│   │   ├── knowledge_base.py    # PubMed + DermNet ingestion
│   │   ├── embeddings.py        # Embedding pipeline
│   │   └── retriever.py         # Pinecone retrieval chain
│   ├── services/
│   │   ├── maps_service.py      # Google Maps Places API
│   │   ├── appointment_service.py
│   │   ├── notification_service.py  # Twilio + SendGrid
│   │   └── celery_tasks.py      # Background task definitions
│   └── core/
│       ├── config.py            # Environment config
│       ├── database.py          # PostgreSQL connection
│       └── auth.py              # JWT authentication
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Patient dashboard
│   │   │   ├── ReportUpload.jsx
│   │   │   ├── SkinAnalysis.jsx
│   │   │   ├── DoctorFinder.jsx
│   │   │   └── DoctorPortal.jsx
│   │   └── components/
│   │       ├── ReportSummary.jsx
│   │       ├── AbnormalityTable.jsx
│   │       ├── BiomarkerChart.jsx
│   │       ├── DoctorCard.jsx
│   │       └── ChatBot.jsx
├── ml/
│   ├── train_skin_classifier.py
│   ├── evaluate_ner.py
│   ├── build_knowledge_base.py
│   └── dvc.yaml                 # DVC pipeline definition
├── mlflow/
│   └── tracking_server/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.vllm
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions pipeline
├── tests/
│   ├── test_ocr.py
│   ├── test_ner.py
│   ├── test_skin_cv.py
│   └── test_rag.py
├── docs/
│   └── medical_ai_platform_architecture.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- GPU with CUDA 12+ (for vLLM — optional, CPU fallback available)
- API keys: OpenAI, Google Maps, Pinecone, Twilio, SendGrid

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/medilens.git
cd medilens
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Fill in your API keys in .env
```

```env
OPENAI_API_KEY=your_key
GOOGLE_MAPS_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENV=your_env
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
SENDGRID_API_KEY=your_key
DATABASE_URL=postgresql://user:password@localhost:5432/medilens
REDIS_URL=redis://localhost:6379
VLLM_BASE_URL=http://localhost:8000
```

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

This starts:
- FastAPI backend on `http://localhost:8080`
- React frontend on `http://localhost:3000`
- PostgreSQL on port `5432`
- Redis on port `6379`
- MLflow UI on `http://localhost:5000`
- ChromaDB on port `8001`

### 4. Run vLLM server (GPU required)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model llava-hf/llava-1.5-7b-hf \
  --port 8000 \
  --dtype float16
```

### 5. Build the knowledge base

```bash
cd ml
python build_knowledge_base.py --sources pubmed dermnet snomed
```

### 6. Run locally without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080

# Frontend
cd frontend
npm install
npm run dev

# Celery worker
celery -A backend.services.celery_tasks worker --loglevel=info
```

---

## Pipeline Walkthrough

### Medical Report Analysis

```python
from backend.graph.state import MedicalState
from backend.graph.builder import build_graph

graph = build_graph()

result = graph.invoke({
    "input_type": "document",
    "raw_input": open("report.pdf", "rb").read(),
    "language": "en"
})

print(result["llm_response"])      # Plain-language explanation
print(result["severity_score"])    # Urgency 1-10
print(result["specialist"])        # Recommended specialty
```

### Skin Image Analysis

```python
result = graph.invoke({
    "input_type": "skin_image",
    "raw_input": open("skin_photo.jpg", "rb").read(),
    "language": "hi"               # Response in Hindi
})

print(result["skin_classification"])  # Condition + confidence
print(result["vllm_description"])     # Clinical description
print(result["severity_score"])       # Urgency score
```

---


---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/report/analyze` | Upload and analyze a medical report |
| `POST` | `/api/skin/analyze` | Upload and analyze a skin image |
| `POST` | `/api/query` | Natural language medical query |
| `GET` | `/api/doctors/nearby` | Find nearby doctors by specialty + location |
| `POST` | `/api/appointment/book` | Book an appointment |
| `GET` | `/api/patient/history` | Patient report history + trends |
| `POST` | `/api/doctor/validate` | Doctor validates / overrides AI analysis |
| `GET` | `/api/health` | Health check |

---

## Disclaimer

> MediLens is a portfolio and research project. All AI-generated outputs are for informational purposes only and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional.

---



## Author

Built by a passionate AI/ML engineer as a production-grade portfolio project demonstrating end-to-end multimodal AI system design.

[![GitHub](https://img.shields.io/badge/GitHub-yourusername-black?style=flat-square&logo=github)](https://github.com/yourusername)
