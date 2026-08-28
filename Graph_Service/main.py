import os
import requests
import traceback
from typing import List, Dict, Any, TypedDict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.auth
from google.oauth2 import service_account
from google import genai
from langgraph.graph import StateGraph, END

from PromptSchema import get_planner_prompts
from ToolsSchema import AutonomousExecutionBlueprint

DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT").strip()
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION").strip()
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
MODEL_NAME = "gemini-2.5-flash-lite"

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

if os.path.exists(KEY_PATH):
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
else:
    credentials, _ = google.auth.default(scopes=SCOPES)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)

def fetch_last_5_conversations(conv_id: str) -> List[Dict[str, Any]]:
    url = f"{DATABASE_SERVICE_URL}/history/{conv_id}?limit=5"
    try:
        response = requests.get(url, timeout=3.0)
        if response.status_code == 200:
            return response.json().get("messages", [])
        return []
    except Exception:
        return []

class GraphState(TypedDict):
    question: str
    conv_id: Optional[str]
    chat_history: List[Dict[str, Any]]
    blueprint: Optional[Dict[str, Any]]

def planner_agent_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    history = state.get("chat_history", [])

    history_str = "\n".join([f"{msg['role']}: {msg['message']}" for msg in history]) if history else "No previous history"

    system_instruction, user_prompt_payload = get_planner_prompts(question, history)

    try:
        response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt_payload,
        config = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": AutonomousExecutionBlueprint,
            "temperature": 0.0,
            "thinking_config": {"thinking_budget": 0}  
        }
    )

        blueprint_data = AutonomousExecutionBlueprint.model_validate_json(response.text)
        return {"blueprint": blueprint_data.model_dump()}
    except Exception as e:
        print("[CRITICAL ERROR IN PLANNER]:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")

workflow = StateGraph(GraphState)
workflow.add_node("planner", planner_agent_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", END)

app_graph = workflow.compile()

app = FastAPI(title="Graph Service", version="1.0.0")

class ProcessQueryRequest(BaseModel):
    question: str
    conv_id: Optional[str] = None

@app.post("/process")
def process_query(payload: ProcessQueryRequest):
    history = []
    if payload.conv_id:
        history = fetch_last_5_conversations(payload.conv_id)

    initial_state: GraphState = {
        "question": payload.question,
        "conv_id": payload.conv_id,
        "chat_history": history,
        "blueprint": None
    }

    final_state = app_graph.invoke(initial_state)

    return final_state["blueprint"]