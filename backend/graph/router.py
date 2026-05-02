"""
backend/graph/router.py

Router Node — the FIRST node in the LangGraph pipeline.

WHAT IT DOES:
  1. Reads the 'input_type' from state
  2. Validates that the input data is present and correct
  3. Returns the state — LangGraph then uses conditional edges to 
     route to the correct pipeline

ROUTING LOGIC (handled by conditional edges in builder.py):
  - input_type="document"   → OCR pipeline (report analysis)
  - input_type="skin_image" → Skin CV pipeline (skin diagnosis)
  - input_type="query"      → Query pipeline (chatbot / symptom checker)
"""

from backend.graph.state import MedicalState


def router_node(state: MedicalState) -> MedicalState:
    """
    Validates the input and prepares it for routing.
    
    The actual routing decision happens via conditional edges in builder.py.
    This node just validates and sets up defaults.
    """
    input_type = state.get("input_type", "")
    
    # Track which nodes we visit (useful for debugging)
    pipeline_path = state.get("pipeline_path", [])
    pipeline_path.append("router")
    
    # Set default language if not provided
    language = state.get("language", "en")
    
    # Validate input_type
    valid_types = {"document", "skin_image", "query"}
    if input_type not in valid_types:
        return {
            **state,
            "error": f"Invalid input_type: '{input_type}'. Must be one of: {valid_types}",
            "pipeline_path": pipeline_path,
            "language": language,
        }
    
    # Validate that we have actual input data
    raw_input = state.get("raw_input")
    query_text = state.get("query_text")
    
    if input_type in ("document", "skin_image") and not raw_input:
        return {
            **state,
            "error": f"input_type is '{input_type}' but no raw_input (file bytes) provided.",
            "pipeline_path": pipeline_path,
            "language": language,
        }
    
    if input_type == "query" and not query_text:
        return {
            **state,
            "error": "input_type is 'query' but no query_text provided.",
            "pipeline_path": pipeline_path,
            "language": language,
        }
    
    print(f"[Router] Input type: {input_type} | Language: {language}")
    
    return {
        **state,
        "pipeline_path": pipeline_path,
        "language": language,
    }


def route_by_input_type(state: MedicalState) -> str:
    """
    Conditional edge function — tells LangGraph which node to go to next.
    
    This function is used in builder.py like:
        graph.add_conditional_edges("router", route_by_input_type, {
            "document":   "ocr_node",
            "skin_image": "skin_cv_node",
            "query":      "query_node",
            "error":      "output_node",
        })
    
    Returns a STRING key that maps to the next node.
    """
    # If there's an error, skip to output
    if state.get("error"):
        return "error"
    
    return state.get("input_type", "error")
