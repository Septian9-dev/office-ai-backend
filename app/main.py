from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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

@app.get("/messages/{agent_id}")
def get_messages(agent_id: str):
    response = supabase.table("messages").select("*").eq("agent_id", agent_id).order("created_at", desc=False).execute()
    return {"messages": response.data}

@app.post("/chat/agent")
async def chat_with_agent(req: ChatRequest):
    msg_lower = req.message.lower()

    briefing_keywords = [
        "briefing", "semua divisi", "lintas divisi", "kumpulkan manajer", 
        "kumpulkan manager", "rapat divisi", "koordinasi divisi", 
        "instruksikan semua", "perintah ke semua", "arahkan", "minta",
        "siapkan konsep", "estimasi", "perbarui", "modern", "proyek baru", "projek baru"
    ]
    is_briefing_kw = any(kw in msg_lower for kw in briefing_keywords)

    # JIKA INSTRUKSI CEO ATAU MEMILIKI KATA KUNCI DELEGASI
    if req.agent_id == "ceo-main" or is_briefing_kw:
        # Simpan pesan pengguna ke riwayat CEO dan agen yang sedang dipilih
        save_message("ceo-main", "You", req.message)
        if req.agent_id != "ceo-main":
            save_message(req.agent_id, "You", req.message)

        result = await run_ceo_briefing(req.message)
        
        involved_ids = ["ceo-main"]
        if isinstance(result, dict) and "error" in result:
            reply_text = f"Maaf, terjadi kesalahan saat menyusun briefing: {result['error']}"
        else:
            reply_text = result.get("master_report", "Gagal memproses briefing CEO.")
            
            # Duplikasi riwayat instruksi & hasil kerja ke agen spesialis yang ditugaskan
            if isinstance(result, dict) and "division_details" in result:
                for div in result["division_details"]:
                    if isinstance(div, dict) and "team_contributions" in div:
                        for contrib in div["team_contributions"]:
                            target_id = contrib.get("agent_id")
                            task_text = contrib.get("task")
                            res_text = contrib.get("result")
                            if target_id:
                                involved_ids.append(target_id)
                                save_message(target_id, "You", f"📌 [Instruksi CEO]: {task_text}")
                                save_message(target_id, contrib.get("agent_name", "Specialist"), res_text)

        save_message("ceo-main", "Pak Pakar (CEO)", reply_text)
        if req.agent_id != "ceo-main":
            save_message(req.agent_id, "Pak Pakar (CEO)", reply_text)

        return {
            "agent_id": "ceo-main",
            "agent_name": "Pak Pakar (CEO)",
            "response": reply_text,
            "involved_ids": list(set(involved_ids))
        }

    # CHAT ORDINARY (1-ON-1)
    save_message(req.agent_id, "You", req.message)
    result = await run_agent_chat(req.agent_id, req.message)
    reply_text = result.get("response", "Tidak ada respon.")
    save_message(req.agent_id, result.get("agent_name", "Agent"), reply_text)
    
    return result

@app.post("/chat/briefing")
async def team_briefing_endpoint(req: BriefingRequest):
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