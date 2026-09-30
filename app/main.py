from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent_engine import run_agent_chat, run_team_briefing, run_ceo_briefing
from app.database import supabase

app = FastAPI(title="3D Office AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    agent_id: str
    message: str

class BriefingRequest(BaseModel):
    division: str
    brief: str

class CEOBriefingRequest(BaseModel):
    brief: str

# Helper untuk menyimpan pesan ke tabel messages di Supabase
def save_message(agent_id: str, sender: str, text: str):
    try:
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": sender,
            "text": text
        }).execute()
    except Exception as e:
        print(f"Error saving message: {e}")

@app.get("/")
def read_root():
    return {"status": "Office AI Backend is running smoothly!"}

@app.get("/agents")
def get_all_agents():
    response = supabase.table("agents").select("id, name, division, role, position_x, position_y, position_z, status").execute()
    return {"agents": response.data}

# Endpoint untuk mengambil riwayat chat agen dari Supabase
@app.get("/messages/{agent_id}")
def get_messages(agent_id: str):
    response = supabase.table("messages").select("*").eq("agent_id", agent_id).order("created_at", desc=False).execute()
    return {"messages": response.data}

@app.post("/chat/agent")
async def chat_with_agent(req: ChatRequest):
    save_message(req.agent_id, "You", req.message)
    result = await run_agent_chat(req.agent_id, req.message)
    
    reply_text = result.get("response", "Tidak ada respon.")
    save_message(req.agent_id, result.get("agent_name", "Agent"), reply_text)
    
    return result

@app.post("/chat/briefing")
async def team_briefing_endpoint(req: BriefingRequest):
    # Ambil ID Manager dari divisi
    res = supabase.table("agents").select("id, name").eq("division", req.division).eq("role", "Manager").execute()
    manager = res.data[0] if res.data else None
    manager_id = manager["id"] if manager else "unknown"
    manager_name = manager["name"] if manager else "Manager"

    save_message(manager_id, "You", req.brief)
    result = await run_team_briefing(req.division, req.brief)
    
    summary_text = result.get("executive_summary", "Gagal memproses briefing tim.")
    save_message(manager_id, f"{manager_name} (Executive Summary)", summary_text)
    
    return result

@app.post("/chat/ceo-briefing")
async def ceo_briefing_endpoint(req: CEOBriefingRequest):
    ceo_id = "ceo-main"
    save_message(ceo_id, "You", req.brief)
    result = await run_ceo_briefing(req.brief)
    
    master_report = result.get("master_report", "Gagal memproses briefing CEO.")
    save_message(ceo_id, "Pak Pakar (CEO) (Master Executive Strategy)", master_report)
    
    return result

@app.websocket("/ws/chat/{agent_id}")
async def websocket_chat(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await run_agent_chat(agent_id, data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print(f"Agent {agent_id} disconnected.")