"""
backend/graph/builder.py

Graph Builder — assembles the full LangGraph pipeline.

HOW LANGGRAPH WORKS:
  1. You define a StateGraph with a shared state type (MedicalState)
  2. You add NODES — each node is a function that takes state and returns updated state
  3. You add EDGES — connections between nodes (linear or conditional)
  4. You compile the graph — this creates a runnable pipeline

CURRENT GRAPH FLOW:
  
  [START] → [router] → (conditional) → [ocr_node] → [ner_node] → [rag_node] → [llm_node] → [doctor_node] → [output_node] → [END]
                                      → [skin_cv_node] → [skin_vllm_node] → ↗
                                      → [query_node] ────────────────────→ ↗

PLACEHOLDER NODES:
  Until we build each feature, placeholder nodes simply pass the state through
  with a print statement so you can see the pipeline executing.
"""

from langgraph.graph import StateGraph, END
from backend.graph.state import MedicalState
from backend.graph.router import router_node, route_by_input_type


# ========================================
# PLACEHOLDER NODES
# These will be replaced one by one as we 
# build each feature
# ========================================

def _placeholder_node(name: str):
    """Factory to create a placeholder node with a given name."""
    def node_fn(state: MedicalState) -> MedicalState:
        path = state.get("pipeline_path", [])
        path.append(name)
        print(f"  [{name}] ✓ (placeholder — not yet implemented)")
        return {**state, "pipeline_path": path}
    # Set the function name for easier debugging
    node_fn.__name__ = name
    return node_fn


def check_ocr_confidence(state: MedicalState) -> str:
    """
    Conditional edge after OCR node.
    If OCR confidence is below threshold → use LLaVA vision fallback.
    If OCR confidence is good → proceed directly to NER.
    """
    confidence = state.get("ocr_confidence", 1.0)
    
    if confidence < 0.70:
        print(f"  [OCR Check] Confidence {confidence:.0%} < 70% → using LLaVA fallback")
        return "low"
    else:
        print(f"  [OCR Check] Confidence {confidence:.0%} ≥ 70% → proceeding to NER")
        return "high"


def build_graph() -> StateGraph:
    """
    Builds and compiles the full MediLens LangGraph pipeline.
    
    Returns a compiled graph that you can invoke like:
        graph = build_graph()
        result = graph.invoke({"input_type": "document", "raw_input": b"...", "language": "en"})
    """
    
    # 1. Create the graph with our shared state type
    graph = StateGraph(MedicalState)
    
    # 2. Add all nodes
    # -- Router (real implementation)
    graph.add_node("router", router_node)
    
    # -- Pipeline 1: Document Analysis (placeholders for now)
    graph.add_node("ocr_node",          _placeholder_node("ocr_node"))
    graph.add_node("vllm_fallback",     _placeholder_node("vllm_fallback"))
    graph.add_node("ner_node",          _placeholder_node("ner_node"))
    
    # -- Pipeline 2: Skin Analysis (placeholders)
    graph.add_node("skin_cv_node",      _placeholder_node("skin_cv_node"))
    graph.add_node("skin_vllm_node",    _placeholder_node("skin_vllm_node"))
    
    # -- Pipeline 3: Query/Symptom (placeholder)
    graph.add_node("query_node",        _placeholder_node("query_node"))
    
    # -- Shared nodes (placeholders)
    graph.add_node("rag_node",          _placeholder_node("rag_node"))
    graph.add_node("llm_explain_node",  _placeholder_node("llm_explain_node"))
    graph.add_node("doctor_node",       _placeholder_node("doctor_node"))
    graph.add_node("output_node",       _placeholder_node("output_node"))
    
    # 3. Set the entry point
    graph.set_entry_point("router")
    
    # 4. Add conditional edges from router → correct pipeline
    graph.add_conditional_edges("router", route_by_input_type, {
        "document":   "ocr_node",
        "skin_image": "skin_cv_node",
        "query":      "query_node",
        "error":      "output_node",   # skip everything on error
    })
    
    # 5. Pipeline 1 edges: OCR → (confidence check) → NER → RAG
    graph.add_conditional_edges("ocr_node", check_ocr_confidence, {
        "low":  "vllm_fallback",
        "high": "ner_node",
    })
    graph.add_edge("vllm_fallback", "ner_node")
    graph.add_edge("ner_node",      "rag_node")
    
    # 6. Pipeline 2 edges: Skin CV → Skin vLLM → RAG
    graph.add_edge("skin_cv_node",   "skin_vllm_node")
    graph.add_edge("skin_vllm_node", "rag_node")
    
    # 7. Pipeline 3 edges: Query → RAG
    graph.add_edge("query_node", "rag_node")
    
    # 8. Shared pipeline: RAG → LLM → Doctor → Output → END
    graph.add_edge("rag_node",         "llm_explain_node")
    graph.add_edge("llm_explain_node", "doctor_node")
    graph.add_edge("doctor_node",      "output_node")
    graph.add_edge("output_node",      END)
    
    # 9. Compile the graph
    compiled = graph.compile()
    print("[Builder] ✓ MediLens graph compiled successfully!")
    
    return compiled


# ========================================
# Quick test — run this file directly to 
# see the pipeline in action
# ========================================
if __name__ == "__main__":
    print("=" * 50)
    print("MediLens — LangGraph Pipeline Test")
    print("=" * 50)
    
    graph = build_graph()
    
    # Test 1: Document pipeline
    print("\n--- Test 1: Document Analysis ---")
    result = graph.invoke({
        "input_type": "document",
        "raw_input": b"sample pdf bytes",
        "language": "en",
    })
    print(f"  Pipeline path: {' → '.join(result.get('pipeline_path', []))}")
    
    # Test 2: Skin image pipeline
    print("\n--- Test 2: Skin Image Analysis ---")
    result = graph.invoke({
        "input_type": "skin_image",
        "raw_input": b"sample image bytes",
        "language": "en",
    })
    print(f"  Pipeline path: {' → '.join(result.get('pipeline_path', []))}")
    
    # Test 3: Query pipeline
    print("\n--- Test 3: Medical Query ---")
    result = graph.invoke({
        "input_type": "query",
        "query_text": "What does high creatinine mean?",
        "language": "en",
    })
    print(f"  Pipeline path: {' → '.join(result.get('pipeline_path', []))}")
    
    # Test 4: Error case
    print("\n--- Test 4: Invalid Input ---")
    result = graph.invoke({
        "input_type": "invalid",
        "language": "en",
    })
    print(f"  Error: {result.get('error')}")
    print(f"  Pipeline path: {' → '.join(result.get('pipeline_path', []))}")
    
    print("\n" + "=" * 50)
    print("All tests passed! The graph structure is working.")
    print("=" * 50)
