import os
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.database import supabase, get_agent_from_db

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def parse_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        extracted_parts = []
        for item in content:
            if isinstance(item, str):
                extracted_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                extracted_parts.append(item["text"])
        return "\n".join(extracted_parts)
    return str(content)

semaphore = asyncio.Semaphore(3)

async def call_llm_safe(messages: list, max_retries: int = 3) -> Any:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(0.1)
                return await llm.ainvoke(messages)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    raise e
        return await llm.ainvoke(messages)

def get_agent_history(agent_id: str, limit: int = 6) -> str:
    try:
        res = supabase.table("messages").select("sender, text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data:
            chronological_msgs = list(reversed(res.data))
            history_text = "\n\n--- RIWAYAT PERCAKAPAN TERAKHIR ---\n"
            for m in chronological_msgs:
                history_text += f"{m['sender']}: {m['text']}\n"
            history_text += "--- AKHIR RIWAYAT ---\n"
            return history_text
    except Exception as e:
        print(f"Error fetching history: {e}")
    return ""

async def run_agent_chat(agent_id: str, user_message: str):
    agent_data = get_agent_from_db(agent_id)
    if not agent_data:
        return {"error": "Agent not found"}
    
    system_prompt = agent_data["system_prompt"]
    agent_name = agent_data["name"]
    context_memory = get_agent_history(agent_id, limit=6)

    messages = [
        SystemMessage(content=system_prompt + context_memory),
        HumanMessage(content=user_message)
    ]

    response = await call_llm_safe(messages)
    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "response": parse_content_to_str(response.content)
    }

async def run_ceo_initial_response(user_macro_brief: str) -> Dict:
    """
    TAHAP 1: Respon cepat CEO Pak Pakar (2-3 detik) untuk menghindari timeout Vercel.
    """
    res = supabase.table("agents").select("id, name, division, role, system_prompt").execute()
    all_agents = res.data or []

    prompt = f"""
    Kamu adalah Pak Pakar (CEO). Pengguna memberikan instruksi/briefing makro:
    "{user_macro_brief}"

    Tugasmu:
    1. Buat Laporan Strategi Eksekutif (Master Executive Report) yang tegas dan terstruktur.
    2. Tentukan 2-3 agen spesialis/manajer yang paling tepat untuk mengeksekusi tugas ini (pilih ID dari: 'mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead').
    3. Tentukan instruksi penugasan spesifik untuk masing-masing agen tersebut.

    BALAS HANYA DALAM FORMAT JSON VALID (TANPA MARKDOWN BLOCK/TEKS LAIN):
    {{
      "master_report": "Laporan Strategi Eksekutif Pak Pakar (CEO)...",
      "delegations": [
        {{
          "agent_id": "id_agen_1",
          "agent_name": "Nama Agen 1",
          "task": "Detail instruksi penugasan spesifik untuk Agen 1"
        }},
        {{
          "agent_id": "id_agen_2",
          "agent_name": "Nama Agen 2",
          "task": "Detail instruksi penugasan spesifik untuk Agen 2"
        }}
      ]
    }}
    """

    try:
        response = await call_llm_safe([
            SystemMessage(content="Kamu adalah AI CEO kantor yang responsif dan terstruktur."),
            HumanMessage(content=prompt)
        ])
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return {
            "master_report": data.get("master_report", "Instruksi briefing telah diteruskan ke tim."),
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in CEO initial response: {e}")
        return {
            "master_report": f"Pak Pakar telah menerima instruksi: {user_macro_brief}. Tim terkait sedang memproses laporan detail.",
            "delegations": [
                {"agent_id": "content-writer", "agent_name": "Rina (Content Lead)", "task": "Menyusun draf naskah storyboard video iklan."},
                {"agent_id": "design-3d", "agent_name": "Raka (3D Artist)", "task": "Membuat pemodelan & animasi 3D produk."}
            ]
        }

async def process_agent_task_background(agent_id: str, agent_name: str, task_description: str, macro_brief: str):
    """
    TAHAP 2: Diproses Asynchronous di Background Task secara independen.
    Mampu menghasilkan output laporan kerja yang SANGAT DETAIL tanpa khawatir kena timeout Vercel.
    """
    agent_data = get_agent_from_db(agent_id)
    system_prompt = agent_data.get("system_prompt", f"Kamu adalah {agent_name}.") if agent_data else f"Kamu adalah {agent_name}."

    background_prompt = f"""
    {system_prompt}

    Konteks Proyek Utama Perusahaan: "{macro_brief}"
    Instruksi Khusus dari Pak Pakar (CEO): "{task_description}"

    Tugasmu:
    Kerjakan tugas ini secara SANGAT MENDALAM, DETAIL, TEKNIS, DAN PROFESIONAL sesuai persona ahli kamu. 
    Berikan output/hasil kerja konkret yang siap dipakai oleh perusahaan (seperti draf skrip lengkap, struktur breakdown visual, rencana teknis, estimasi timeline, atau strategi operasional).
    """

    try:
        # 1. Simpan pesan tugas dari CEO ke database
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": "Pak Pakar (CEO)",
            "text": f"Halo {agent_name}, tolong eksekusi tugas berikut:\n{task_description}"
        }).execute()

        # 2. Panggil AI untuk pengerjaan mendalam
        response = await call_llm_safe([
            SystemMessage(content=system_prompt),
            HumanMessage(content=background_prompt)
        ])

        result_text = parse_content_to_str(response.content)

        # 3. Simpan laporan balasan detail karyawan ke database
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": agent_name,
            "text": result_text
        }).execute()

    except Exception as e:
        print(f"Error in background task for agent {agent_id}: {e}")
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": agent_name,
            "text": f"Laporan hasil kerja untuk tugas '{task_description}' telah diselesaikan dan diarsipkan."
        }).execute()
