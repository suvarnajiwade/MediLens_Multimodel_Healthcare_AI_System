"""
backend/graph/state.py

MedicalState — the shared state that flows through every node in the LangGraph pipeline.

HOW IT WORKS:
- LangGraph passes this state dict from node to node.
- Each node reads what it needs, does its work, and writes its output back to the state.
- Think of it like a medical chart that every department (node) fills in.

EXAMPLE FLOW:
  1. Router reads 'input_type' → decides which pipeline to enter
  2. OCR node reads 'raw_input' → writes 'ocr_text' + 'ocr_confidence'
  3. NER node reads 'ocr_text' → writes 'ner_entities' + 'abnormalities'
  4. RAG node reads entities → writes 'rag_context'
  5. LLM node reads everything → writes 'llm_response' + 'severity_score'
"""

from typing import TypedDict, Optional


class MedicalState(TypedDict, total=False):
    """
    Shared state for the MediLens LangGraph pipeline.
    
    'total=False' means all fields are optional — each node only fills
    in the fields it's responsible for. This prevents errors when the
    pipeline starts with mostly empty state.
    """

    # ===========================
    # INPUT (set by the user/API)
    # ===========================
    input_type: str              # "document" | "skin_image" | "query"
    raw_input: bytes             # raw file bytes (PDF, image) or text query as bytes
    query_text: str              # for text queries (Pipeline 3)
    language: str                # user's preferred language (default: "en")

    # ===========================
    # OCR OUTPUT (Pipeline 1)
    # ===========================
    ocr_text: str                # extracted text from PDF/image
    ocr_confidence: float        # 0.0 to 1.0 — if below 0.70, triggers vLLM fallback

    # ===========================
    # NER OUTPUT (Pipeline 1)
    # ===========================
    ner_entities: dict            # extracted medical entities
    # Example structure:
    # {
    #     "biomarkers": [{"name": "Hemoglobin", "value": "9.2", "unit": "g/dL"}],
    #     "medications": ["Metformin 500mg"],
    #     "diagnoses": ["Type 2 Diabetes"],
    #     "anatomical": ["Liver", "Kidney"]
    # }

    abnormalities: list           # flagged abnormal values
    # Example:
    # [
    #     {"name": "Creatinine", "value": 1.8, "ref_range": "0.6-1.2",
    #      "unit": "mg/dL", "status": "ABNORMAL"}
    # ]

    # ===========================
    # SKIN CV OUTPUT (Pipeline 2)
    # ===========================
    skin_classification: dict     # CV model output
    # Example:
    # {"label": "Psoriasis", "confidence": 0.87, "all_classes": {...}}

    vllm_description: str         # LLaVA clinical description of skin image

    # ===========================
    # QUERY OUTPUT (Pipeline 3)
    # ===========================
    query_intent: str             # classified intent type
    # One of: REPORT_QUESTION | SYMPTOM_DESCRIPTION | SPECIALIST_SEARCH
    #         | APPOINTMENT_REQUEST | GENERAL_MEDICAL_QA

    extracted_symptoms: list      # symptoms found in query text

    # ===========================
    # RAG OUTPUT (Shared Layer)
    # ===========================
    rag_context: list             # retrieved documents from knowledge base
    rag_query: str                # the query sent to the retriever

    # ===========================
    # LLM OUTPUT (Final Analysis)
    # ===========================
    llm_response: str             # plain-language medical explanation
    severity_score: int           # urgency score 1-10
    specialist: str               # recommended specialist (e.g. "Cardiologist")

    # ===========================
    # DOCTOR & APPOINTMENT
    # ===========================
    nearby_doctors: list          # list of doctor dicts from Maps API
    appointment_slot: dict        # booked appointment details

    # ===========================
    # METADATA
    # ===========================
    error: str                    # error message if something fails
    pipeline_path: list           # tracks which nodes were visited (for debugging)
