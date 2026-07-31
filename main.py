from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from dynamic_workflow import create_dynamic_workflow
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from utils.knowledge import PersistentHybridKB, register_existing_kb, get_agent_kb
import traceback
import logging

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Discvr AI Studio Engine")

# CORS Setup for Frontend communication
# Note: allow_origins=["*"] cannot be combined with allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentConfig(BaseModel):
    id: str
    name: str
    system_prompt: str
    tools: List[str] = []
    temperature: float = 0
    guardrails: Optional[str] = None
    knowledge_base: Optional[str] = None
    kb_type: str = "markdown"  # "markdown", "vector", or "remote"
    kb_provider: Optional[str] = None # "pinecone", "qdrant", "rest_api"
    kb_url: Optional[str] = None
    kb_api_key: Optional[str] = None
    kb_webhook_url: Optional[str] = None # For n8n
    kb_embedding_model: str = "openai-3-small" # "openai-3-small", "bedrock-titan", etc.
    kb_dimensions: int = 1536
    webhook_url: Optional[str] = None
    downstream_nodes: List[str] = [] # For explicit routing

    @field_validator("id")
    @classmethod
    def id_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent id cannot be empty")
        return v.strip()

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

class WorkflowRequest(BaseModel):
    agents: List[AgentConfig] = Field(..., min_length=1)
    input_text: str = Field(..., min_length=1)
    workflow_mode: str = "sequential" # "sequential", "graph", or "supervisor"

    @field_validator("workflow_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"sequential", "graph", "supervisor"}
        if v not in allowed:
            raise ValueError(f"workflow_mode must be one of: {', '.join(sorted(allowed))}")
        return v

@app.get("/")
def read_root():
    return {"status": "Discvr AI Studio Engine is active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def _serialize_content(content) -> str:
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts)
    if isinstance(content, str):
        return content
    return str(content)

def _message_sender(msg) -> str:
    msg_type = getattr(msg, "type", "")
    if msg_type == "human":
        return "User"
    if msg_type == "tool":
        return getattr(msg, "name", None) or "Tool"
    name = getattr(msg, "name", None)
    if name:
        return name
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    if kwargs.get("agent_name"):
        return kwargs["agent_name"]
    return "Agent"

@app.post("/api/orchestrate")
async def run_dynamic_orchestration(request: WorkflowRequest):
    """
    Creates agents on the fly based on user config from the Studio UI.
    """
    try:
        configs = [a.model_dump() for a in request.agents]
        app_graph = create_dynamic_workflow(configs, workflow_mode=request.workflow_mode)
        
        inputs = {
            "messages": [HumanMessage(content=request.input_text)],
            "extracted_data": {},
            "next": "",
        }
        # Limit recursion so supervisor loops cannot hang the request indefinitely
        result = await app_graph.ainvoke(inputs, config={"recursion_limit": 50})
        
        messages_history = []
        for msg in result.get('messages', []):
            messages_history.append({
                "type": getattr(msg, "type", "unknown"),
                "content": _serialize_content(msg.content),
                "sender": _message_sender(msg),
            })
            
        return {
            "status": "success",
            "result": messages_history,
            "extracted_data": result.get("extracted_data") or {}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/upload")
async def upload_kb_file(agent_id: str = Form(...), file: UploadFile = File(...)):
    """
    Handles binary file uploads (PDF, Docx, Excel) and ingests them into the persistent KB.
    """
    try:
        if not agent_id.strip():
            raise HTTPException(status_code=400, detail="agent_id is required")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Reuse an existing in-memory KB so searches see the new docs immediately
        existing = get_agent_kb(agent_id)
        if isinstance(existing, PersistentHybridKB):
            kb = existing
        else:
            kb = PersistentHybridKB(agent_id)

        success = kb.ingest_binary(content, file.filename)
        
        if success:
            register_existing_kb(agent_id, kb)
            return {"status": "success", "message": f"File '{file.filename}' indexed for agent {agent_id}"}
        raise HTTPException(status_code=400, detail="Failed to parse file or unsupported format.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
