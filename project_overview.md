# Medical AI Platform — Full Pipeline Architecture

> **Stack:** Python · FastAPI · LangChain · LangGraph · vLLM · React · Docker · MLflow · AWS/GCP  
> **Purpose:** End-to-end AI platform for medical report analysis, skin condition diagnosis, specialist recommendation, and appointment booking.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline 1 — Text / Document Report Analysis](#2-pipeline-1--text--document-report-analysis)
3. [Pipeline 2 — Skin Image Analysis](#3-pipeline-2--skin-image-analysis)
4. [Pipeline 3 — Conversational Query & Symptom Checker](#4-pipeline-3--conversational-query--symptom-checker)
5. [Shared Layer — LangChain RAG Pipeline](#5-shared-layer--langchain-rag-pipeline)
6. [LangGraph State Machine — Orchestrator](#6-langgraph-state-machine--orchestrator)
7. [Doctor Finder & Appointment Engine](#7-doctor-finder--appointment-engine)
8. [Output & Dashboard Layer](#8-output--dashboard-layer)
9. [MLOps & Infra Layer](#9-mlops--infra-layer)
10. [Full Tech Stack Reference](#10-full-tech-stack-reference)
11. [Data Flow Summary](#11-data-flow-summary)

---

## 1. System Overview

The platform accepts five types of user input:

| Input Type | Example | Routed To |
|---|---|---|
| PDF / lab report | Blood test PDF | Pipeline 1 — OCR + NER |
| Scanned report image | Photo of a report | Pipeline 1 — OCR → vLLM fallback |
| Skin photo | Image of rash/lesion | Pipeline 2 — CV + vLLM |
| Voice / text query | "What does high creatinine mean?" | Pipeline 3 — LangChain Q&A |
| Symptom description | "I have chest pain and fatigue" | Pipeline 3 — Symptom extractor |

All inputs flow through a **LangGraph state machine** that classifies the input type and routes it to the correct pipeline. All three pipelines converge at a shared **LangChain RAG layer** before producing output.

---

## 2. Pipeline 1 — Text / Document Report Analysis

This pipeline handles PDF reports, typed text, and scanned/photographed document images.

### Stage 1 — Document Ingestion

```
User uploads PDF or image
         ↓
FastAPI endpoint → async Celery task queue
         ↓
File type detection (PDF / image / plain text)
```

- PDFs are parsed using `PyMuPDF` (fitz) to extract raw text and embedded images.
- Plain text is passed directly to Stage 2.
- Images (scanned reports) are passed to the OCR engine.

### Stage 2 — OCR Engine with vLLM Fallback

This is a two-stage OCR pipeline with an automatic quality fallback.

```
Scanned image
      ↓
Stage A: Tesseract OCR + PaddleOCR (ensemble)
      ↓
Confidence score computed (character-level + word-level)
      ↓
  conf >= 70% ?
  ┌────────────┬─────────────────────────────┐
  │ YES        │ NO                          │
  ↓            ↓                             │
Use OCR text   vLLM Vision Model (LLaVA-1.6 / Qwen-VL)
               reads image directly as multimodal input
               returns structured clinical text
               └─────────────────────────────┘
                          ↓
               Merged text output → Stage 3
```

**Why vLLM here:**  
Traditional OCR fails on handwritten prescriptions, low-resolution scans, Tamil/Hindi text in reports, and non-standard report formats. A multimodal vLLM model (served via the vLLM inference engine for high throughput) reads these as images and returns clean extracted text — no preprocessing required.

**LangChain integration:**  
The OCR + fallback decision logic is wrapped as a `LangChain Tool` called `OCRExtractionTool`, invoked by the LangGraph node `ocr_node`.

### Stage 3 — Medical Named Entity Recognition (NER)

```
Raw extracted text
        ↓
BioBERT / scispaCy NER model
        ↓
Entities extracted:
  - Biomarkers      (Hemoglobin: 9.2 g/dL)
  - Medications     (Metformin 500mg)
  - Diagnoses       (Type 2 Diabetes)
  - Anatomical refs (Liver, Kidney)
  - Lab values      (Creatinine: 1.8 mg/dL, ref: 0.6–1.2)
        ↓
Abnormality detection:
  - Each lab value compared to reference range
  - Flagged as: NORMAL / BORDERLINE / ABNORMAL / CRITICAL
        ↓
Urgency score assigned (1–10)
```

**Models used:**
- `en_core_sci_lg` (scispaCy) for general biomedical NER
- Fine-tuned BioBERT on i2b2 / MIMIC-III datasets for clinical entities
- Rule-based range checker using LOINC reference database

**LangChain integration:**  
Wrapped as `MedicalNERTool` — a custom LangChain tool that accepts raw text and returns a structured JSON of entities + abnormality flags.

### Stage 4 → Shared RAG Layer

NER output is passed to the shared RAG pipeline (Section 5).

---

## 3. Pipeline 2 — Skin Image Analysis

This pipeline handles photographs of skin conditions uploaded by users.

### Stage 1 — Image Preprocessing

```
User uploads skin photo
        ↓
Resize to 224x224, normalize (ImageNet stats)
Apply CLAHE for contrast enhancement
Skin region detection (optional — remove background noise)
```

### Stage 2 — Condition Classification

```
Preprocessed image
        ↓
EfficientNet-B4 fine-tuned on HAM10000 + ISIC 2020 dataset
        ↓
Multi-class output:
  - Melanoma
  - Basal cell carcinoma
  - Eczema / Atopic dermatitis
  - Psoriasis
  - Acne (mild / moderate / severe)
  - Rosacea
  - Fungal infection (tinea)
  - Wart / molluscum
  - Normal skin
        ↓
Confidence score per class (softmax probability)
```

**Dataset:**
- HAM10000: 10,015 dermatoscopic images, 7 classes
- ISIC 2020: 33,126 images for melanoma classification
- Both are publicly available on Kaggle / ISIC archive

### Stage 3 — Severity Scoring

```
Classification output + image features
        ↓
Regression head (MLP on top of EfficientNet features)
        ↓
Severity score: 1–10
  1–3 → Mild (self-care + monitor)
  4–6 → Moderate (general physician)
  7–8 → Severe (dermatologist)
  9–10 → Urgent (oncology / emergency referral)
```

### Stage 4 — vLLM Visual Clinical Description

```
Original skin image
        ↓
vLLM multimodal model (LLaVA-1.6 or Qwen-VL-Chat)
served via vLLM inference engine
        ↓
Prompt: "Describe the skin condition in this image using clinical terminology.
         Note color, texture, spread pattern, borders, and any lesion characteristics."
        ↓
Clinical description text generated
  e.g. "Erythematous scaly plaques with well-defined borders on the extensor surface,
        consistent with plaque psoriasis."
```

**Why vLLM for this:**  
The same vLLM server used as OCR fallback (Section 2) is reused here — no additional infrastructure. vLLM's PagedAttention enables high-throughput batched inference for concurrent image requests.

### Stage 5 — Melanoma Risk Flag

```
If class = Melanoma AND confidence > 0.65:
  → Activate ABCDE rule checker (Asymmetry, Border, Color, Diameter, Evolution)
  → Set urgency = 9–10
  → Generate urgent referral flag for oncologist
```

### Stage 6 → Shared RAG Layer

Classification label + severity score + vLLM clinical description → passed to RAG pipeline.

---

## 4. Pipeline 3 — Conversational Query & Symptom Checker

This pipeline handles natural language queries, symptom descriptions, and voice inputs.

### Stage 1 — Input Normalization

```
Voice input → Whisper (OpenAI) STT → text
Text input  → direct
        ↓
LangChain PromptTemplate normalizes input
```

### Stage 2 — Intent Classification

```
Normalized text
        ↓
LangChain LLMChain with intent classifier prompt
        ↓
Intent types:
  - REPORT_QUESTION     ("What does my high uric acid mean?")
  - SYMPTOM_DESCRIPTION ("I have persistent cough and fever")
  - SPECIALIST_SEARCH   ("Find me a cardiologist near me")
  - APPOINTMENT_REQUEST ("Book an appointment for next Monday")
  - GENERAL_MEDICAL_QA  ("What is HbA1c?")
```

### Stage 3 — Symptom Extraction

```
If intent = SYMPTOM_DESCRIPTION:
        ↓
scispaCy NER → extract symptom entities
        ↓
Symptom-to-specialty mapping table:
  chest pain + fatigue → Cardiology
  persistent headache + blurred vision → Neurology
  joint swelling + stiffness → Rheumatology
  blood in urine → Urology / Nephrology
  skin rash (without image) → Dermatology
        ↓
Specialty identified → passed to Doctor Finder module
```

### Stage 4 — LangGraph Routing Decision

```
Intent + extracted entities
        ↓
LangGraph conditional edge:
  REPORT_QUESTION    → RAG Q&A chain
  SYMPTOM_DESC       → Doctor finder + appointment
  SPECIALIST_SEARCH  → Doctor finder
  APPT_REQUEST       → Appointment engine
  GENERAL_QA         → RAG Q&A chain
```

---

## 5. Shared Layer — LangChain RAG Pipeline

All three pipelines converge here. This layer generates the final grounded medical answer.

### Knowledge Base

| Source | Content | Format |
|---|---|---|
| PubMed abstracts | 35M+ biomedical papers | Chunked + embedded |
| SNOMED-CT | Clinical terminology ontology | Structured lookup |
| DermNet NZ | Dermatology condition database | Chunked + embedded |
| MedlinePlus | Patient-friendly health articles | Chunked + embedded |
| LOINC reference ranges | Lab value normal ranges | Lookup table |

### Embedding & Vector Store

```
Knowledge base documents
        ↓
Chunked (512 tokens, 50 overlap) using LangChain TextSplitter
        ↓
Embedded using: text-embedding-ada-002 / BioBERT sentence embeddings
        ↓
Stored in: Pinecone (production) / ChromaDB (development)
```

### Retrieval Chain (LangChain)

```
Query = NER entities + classification label + clinical description
        ↓
LangChain RetrievalQA chain:
  - Similarity search: top-k=5 relevant chunks from vector store
  - Reranker: Cohere reranker / cross-encoder for precision
        ↓
Retrieved context + query → LLM (GPT-4 / Claude / Llama-3 via vLLM)
        ↓
LangChain ConversationBufferMemory maintains multi-turn context
        ↓
Final response:
  - Plain-language explanation of findings
  - What each abnormal value means
  - What specialist is recommended
  - Urgency level with reason
  - Disclaimer: "Consult a licensed physician for diagnosis"
```

### Multilingual Output

```
Detected user language (langdetect library)
        ↓
LLM prompt: "Respond in {language}"
Supported: English, Hindi, Marathi, Tamil, Bengali, Telugu
```

---

## 6. LangGraph State Machine — Orchestrator

LangGraph manages the entire flow as a stateful graph. Each pipeline stage is a **node**, and routing decisions are **conditional edges**.

### Graph Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class MedicalState(TypedDict):
    input_type: str          # "document" | "skin_image" | "query"
    raw_input: bytes | str
    ocr_text: str
    ocr_confidence: float
    ner_entities: dict
    skin_classification: dict
    severity_score: int
    vllm_description: str
    rag_context: list
    llm_response: str
    specialist: str
    urgency: int
    appointment_slot: dict
    language: str

graph = StateGraph(MedicalState)

# Nodes
graph.add_node("router",          router_node)
graph.add_node("ocr_node",        ocr_node)
graph.add_node("vllm_fallback",   vllm_ocr_node)
graph.add_node("ner_node",        ner_node)
graph.add_node("skin_cv_node",    skin_cv_node)
graph.add_node("skin_vllm_node",  skin_vllm_describe_node)
graph.add_node("query_node",      query_intent_node)
graph.add_node("rag_node",        rag_retrieval_node)
graph.add_node("llm_explain_node",llm_explanation_node)
graph.add_node("doctor_node",     doctor_finder_node)
graph.add_node("appointment_node",appointment_node)
graph.add_node("output_node",     output_formatter_node)

# Edges
graph.set_entry_point("router")

graph.add_conditional_edges("router", route_by_input_type, {
    "document":   "ocr_node",
    "skin_image": "skin_cv_node",
    "query":      "query_node"
})

graph.add_conditional_edges("ocr_node", check_ocr_confidence, {
    "low":  "vllm_fallback",
    "high": "ner_node"
})

graph.add_edge("vllm_fallback",   "ner_node")
graph.add_edge("ner_node",        "rag_node")
graph.add_edge("skin_cv_node",    "skin_vllm_node")
graph.add_edge("skin_vllm_node",  "rag_node")
graph.add_edge("query_node",      "rag_node")
graph.add_edge("rag_node",        "llm_explain_node")
graph.add_edge("llm_explain_node","doctor_node")
graph.add_edge("doctor_node",     "appointment_node")
graph.add_edge("appointment_node","output_node")
graph.add_edge("output_node",     END)

app = graph.compile()
```

### Key LangGraph Benefits Used

- **Persistent state** across all nodes — no need to pass context manually
- **Conditional edges** for OCR confidence check and input type routing
- **Human-in-the-loop** support — doctor review node can pause graph and await doctor validation before output
- **Streaming** support — intermediate results streamed to frontend as each node completes

---

## 7. Doctor Finder & Appointment Engine

### Doctor Finder

```
Specialist type + user GPS coordinates
        ↓
Google Maps Places API:
  query: "cardiologist near {lat},{lng}"
  radius: user-defined (5km / 10km / 25km)
        ↓
Results filtered by:
  - Rating >= 4.0
  - Verified clinic
  - Accepts relevant insurance (optional)
  - Language spoken (optional)
        ↓
AI recommendation explanation:
  LangChain prompt: "Given these report findings: {findings},
  explain in 2 sentences why Dr. {name} ({specialty}) is recommended."
        ↓
Ranked doctor list returned to frontend
```

### Appointment Engine

```
Selected doctor + urgency score
        ↓
Fetch available slots (doctor calendar API / mock)
        ↓
Urgency-based slot selection:
  urgency 8–10 → earliest available slot (within 24 hrs)
  urgency 4–7  → next available within 3 days
  urgency 1–3  → standard booking (any available)
        ↓
Slot confirmed:
  - Stored in PostgreSQL appointments table
  - Celery task triggered: send SMS (Twilio) + email (SendGrid)
  - Reminder task scheduled: 24hrs before + 1hr before
        ↓
Appointment confirmation returned to frontend
```

---

## 8. Output & Dashboard Layer

### Patient Dashboard (React)

| Component | Data Source |
|---|---|
| Report summary card | LLM explanation output |
| Abnormal values table | NER + range checker |
| Severity badge | Urgency scorer |
| Skin condition result | CV pipeline + vLLM description |
| Biomarker trend chart | Historical report comparison |
| Recommended specialist | Specialist recommender |
| Nearby doctors map | Google Maps embedded |
| Booked appointment | Appointment engine |
| Report Q&A chat | LangGraph conversational chain |

### Doctor Review Portal (React)

| Component | Purpose |
|---|---|
| AI analysis panel | Shows AI-generated summary |
| Override / Validate button | Doctor corrects AI output (HITL) |
| Patient history timeline | Past reports + trend graphs |
| Appointment queue | Today's scheduled patients |
| Flagged critical cases | Patients with urgency >= 8 |

---

## 9. MLOps & Infra Layer

### Model Versioning & Tracking

```
MLflow Tracking Server
  - Logs: OCR confidence metrics, NER F1 score, skin CV accuracy, RAG retrieval precision
  - Artifacts: model weights, training configs, confusion matrices
  - Model registry: stage = Staging → Production with approval gate

DVC (Data Version Control)
  - Tracks dataset versions (HAM10000, ISIC, MIMIC-III preprocessed splits)
  - Pipeline steps defined in dvc.yaml
  - Reproducible experiments via dvc repro
```

### Serving Layer

```
vLLM Server (GPU)
  - Serves: LLaVA-1.6 (vision), Llama-3 / Mistral (text LLM)
  - PagedAttention for high-throughput batching
  - OpenAI-compatible API endpoint

FastAPI Backend
  - Async endpoints for all pipeline entry points
  - Celery + Redis for background task queue (OCR, email jobs)
  - PostgreSQL for user data, reports, appointments
  - Pinecone client for vector store queries
```

### Infrastructure

```
Docker Compose (development):
  services: fastapi, celery-worker, redis, postgres, vllm, mlflow, chromadb

Kubernetes (production on AWS EKS / GCP GKE):
  - FastAPI deployment (autoscaling)
  - vLLM deployment on GPU node pool
  - Celery workers deployment
  - MLflow tracking server

CI/CD (GitHub Actions):
  on push → main:
    - Run pytest (unit + integration tests)
    - Build Docker images
    - Push to ECR / GCR
    - Deploy to K8s cluster via kubectl
    - Run smoke tests on staging
    - Promote to production if all pass
```

---

## 10. Full Tech Stack Reference

| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS, React Query |
| API layer | FastAPI (async), Pydantic v2 |
| Task queue | Celery + Redis |
| Orchestration | LangGraph (state machine) |
| LLM framework | LangChain (chains, tools, memory, RAG) |
| LLM inference | vLLM (LLaVA-1.6, Llama-3, Qwen-VL) |
| OCR | Tesseract, PaddleOCR |
| PDF parsing | PyMuPDF (fitz) |
| Medical NER | scispaCy, BioBERT (HuggingFace) |
| Skin CV model | EfficientNet-B4 (PyTorch / timm) |
| Embeddings | BioBERT / text-embedding-ada-002 |
| Vector store | Pinecone (prod), ChromaDB (dev) |
| Database | PostgreSQL (pgvector extension) |
| Speech-to-text | OpenAI Whisper |
| Maps / location | Google Maps Places API |
| Notifications | Twilio (SMS), SendGrid (email) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Containerization | Docker, Docker Compose |
| Cloud | AWS EKS / GCP GKE |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

---

## 11. Data Flow Summary

```
User Input
    │
    ▼
[LangGraph Router] ──────────────────────────────────────┐
    │                                                     │
    │ document                skin_image      query       │
    ▼                              ▼             ▼        │
[OCR Engine]              [EfficientNet-B4]  [Whisper]   │
    │                         + [vLLM]       + [Intent]  │
    │ low conf                                            │
    ▼                                                     │
[vLLM Fallback]                                          │
    │                                                     │
    ▼                                                     │
[Medical NER]                                            │
    │                                                     │
    └──────────────────────┬──────────────────────────────┘
                           ▼
                  [LangChain RAG Pipeline]
                    Pinecone + PubMed + DermNet
                           │
                           ▼
                  [LLM Explainer + Urgency Scorer]
                           │
                           ▼
                  [Specialist Recommender]
                           │
                           ▼
                  [Doctor Finder + Appointment Engine]
                           │
                           ▼
                  [Patient Dashboard + Doctor Portal]
                           │
                           ▼
                  [MLflow Feedback Loop → Model Improvement]
```

---

> **Disclaimer:** This platform is for informational and portfolio demonstration purposes. All AI outputs must be reviewed by a licensed medical professional before any clinical use.