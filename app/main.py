from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent_engine import run_agent_chat, run_ceo_initial_response, process_agent_task_background
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
async def chat_with_agent(req: ChatRequest, background_tasks: BackgroundTasks):
    msg_lower = req.message.lower()

    briefing_keywords = [
        "briefing", "semua divisi", "lintas divisi", "kumpulkan manajer", 
        "kumpulkan manager", "rapat divisi", "koordinasi divisi", 
        "instruksikan semua", "perintah ke semua", "arahkan", "minta",
        "siapkan konsep", "estimasi", "perbarui", "modern", "proyek baru", 
        "projek baru", "video", "iklan", "buatkan video"
    ]
    is_briefing_kw = any(kw in msg_lower for kw in briefing_keywords)

    if req.agent_id == "ceo-main" or is_briefing_kw:
        # 1. Simpan pertanyaan user
        save_message("ceo-main", "You", req.message)
        if req.agent_id != "ceo-main":
            save_message(req.agent_id, "You", req.message)

        # 2. Ambil respons eksekutif cepat dari CEO (2-3 detik)
        ceo_result = await run_ceo_initial_response(req.message)
        master_report = ceo_result.get("master_report", "Laporan briefing diteruskan.")
        delegations = ceo_result.get("delegations", [])

        save_message("ceo-main", "Pak Pakar (CEO)", master_report)

        involved_ids = ["ceo-main"]

        # 3. Daftarkan pengerjaan mendalam tiap karyawan ke Background Tasks
        for item in delegations:
            target_id = item.get("agent_id")
            target_name = item.get("agent_name", "Spesialis")
            task_text = item.get("task", "")

            if target_id:
                involved_ids.append(target_id)
                background_tasks.add_task(
                    process_agent_task_background,
                    target_id,
                    target_name,
                    task_text,
                    req.message
                )

        return {
            "agent_id": "ceo-main",
            "agent_name": "Pak Pakar (CEO)",
            "response": master_report,
            "delegations": delegations,
            "involved_ids": list(set(involved_ids))
        }

    # Chat biasa 1-on-1 dengan agen
    save_message(req.agent_id, "You", req.message)
    result = await run_agent_chat(req.agent_id, req.message)
    reply_text = result.get("response", "Tidak ada respon.")
    save_message(req.agent_id, result.get("agent_name", "Agent"), reply_text)
    
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
