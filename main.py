from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from dynamic_workflow import create_dynamic_workflow
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from utils.knowledge import PersistentHybridKB, register_agent_kb
import traceback

load_dotenv()

app = FastAPI(title="Discvr AI Studio Engine")

# CORS Setup for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For POC development
    allow_credentials=True,
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

class WorkflowRequest(BaseModel):
    agents: List[AgentConfig]
    input_text: str
    workflow_mode: str = "sequential" # "sequential" or "supervisor"

@app.get("/")
def read_root():
    return {"status": "Discvr AI Studio Engine is active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/orchestrate")
async def run_dynamic_orchestration(request: WorkflowRequest):
    """
    Creates agents on the fly based on user config from the Studio UI.
    """
    try:
        # 1. Build the graph dynamically based on UI configuration
        configs = [a.model_dump() for a in request.agents]
        app_graph = create_dynamic_workflow(configs, workflow_mode=request.workflow_mode)
        
        # 2. Execute the workflow
        # Initial state includes the user's input message
        inputs = {"messages": [HumanMessage(content=request.input_text)]}
        result = await app_graph.ainvoke(inputs)
        
        # 3. Serialize messages for frontend display
        messages_history = []
        for msg in result.get('messages', []):
            content = msg.content
            # Handle cases where content is a list of blocks
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)
            
            # Identify sender
            msg_type = msg.type
            sender = "User" if msg_type == "human" else "Agent"
            
            # Attempt to find which agent sent it (metadata or additional logic could be here)
            # For this POC, we'll label AI messages clearly
            
            messages_history.append({
                "type": msg_type,
                "content": content,
                "sender": sender
            })
            
        return {
            "status": "success",
            "result": messages_history,
            "extracted_data": result.get("extracted_data")
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/upload")
async def upload_kb_file(agent_id: str = Form(...), file: UploadFile = File(...)):
    """
    Handles binary file uploads (PDF, Docx, Excel) and ingests them into the persistent KB.
    """
    try:
        content = await file.read()
        kb = PersistentHybridKB(agent_id)
        success = kb.ingest_binary(content, file.filename)
        
        if success:
            return {"status": "success", "message": f"File '{file.filename}' indexed for agent {agent_id}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to parse file or unsupported format.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
