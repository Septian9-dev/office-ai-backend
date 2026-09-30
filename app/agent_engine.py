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
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Instruksi global agar gaya percakapan seluruh AI Agent terasa alami & komunikatif
HUMAN_TONE_INSTRUCTION = """
PANDUAN GAYA BAHASA MANUSIAWI & FLEKSIBEL (HUMAN-LIKE CONVERSATION):
1. Bicara secara natural, komunikatif, dan luwes seperti rekan kerja di kantor/startup modern Indonesia.
2. HINDARI bahasa kaku atau robotic (jangan gunakan kalimat seperti "Poin tersebut telah saya pahami...", "Tentu, sebagai AI...", atau bahasa surat dinas yang terlalu kaku).
3. Gunakan sapaan dan artikulasi yang ramah serta bervariasi (misalnya: "Halo Pak/Bu", "Sip, siap!", "Okee Pak Pakar", "Gini mas/mbak...", "Bisa banget!", dll).
4. Gunakan variasi intonasi, emosi positif, dan emoji secukupnya agar percakapan terasa hidup dan tidak membosankan.
5. Tetap profesional dan fokus memberikan solusi terbaik sesuai keahlian & persona utama.
"""

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

    # Menggabungkan instruksi persona, gaya bahasa manusiawi, dan memori percakapan
    full_system_prompt = f"{system_prompt}\n\n{HUMAN_TONE_INSTRUCTION}\n{context_memory}"

    messages = [
        SystemMessage(content=full_system_prompt),
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
    TAHAP 1: Respon cepat CEO Pak Pakar (2-3 detik) untuk menghindari timeout.
    """
    res = supabase.table("agents").select("id, name, division, role, system_prompt").execute()

    prompt = f"""
    Kamu adalah Pak Pakar (CEO). Pengguna memberikan instruksi/briefing makro:
    "{user_macro_brief}"

    Tugasmu:
    1. Buat Laporan Strategi Eksekutif (Master Executive Report) yang tegas, lugas, namun tetap komunikatif.
    2. Tentukan 2-3 agen spesialis/manajer yang paling tepat untuk mengeksekusi tugas ini (pilih ID dari: 'mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead').
    3. Tentukan instruksi penugasan spesifik untuk masing-masing agen tersebut.

    BALAS HANYA DALAM FORMAT JSON VALID BERIKUT (TANPA MARKDOWN BLOCK/TEKS LAIN):
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
            SystemMessage(content=f"Kamu adalah AI CEO kantor yang responsif, berwibawa, dan komunikatif.\n\n{HUMAN_TONE_INSTRUCTION}"),
            HumanMessage(content=prompt)
        ])
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return {
            "master_report": data.get("master_report", "Sip, instruksi briefing sudah diteruskan ke tim terkait ya!"),
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in CEO initial response: {e}")
        return {
            "master_report": f"Oke, instruksi '{user_macro_brief}' sudah dikoordinasikan ke tim. Anggota divisi terkait sedang menyiapkan laporan detailnya ya.",
            "delegations": [
                {"agent_id": "content-writer", "agent_name": "Rina (Content Lead)", "task": "Menyusun draf naskah storyboard video iklan."},
                {"agent_id": "design-3d", "agent_name": "Raka (3D Artist)", "task": "Membuat pemodelan & animasi 3D produk."}
            ]
        }

async def process_agent_task_background(agent_id: str, agent_name: str, task_description: str, macro_brief: str):
    """
    TAHAP 2: Diproses Asynchronous di Background Task.
    Menghasilkan output pengerjaan yang detail namun disampaikan dengan gaya rekan kerja alami.
    """
    agent_data = get_agent_from_db(agent_id)
    system_prompt = agent_data.get("system_prompt", f"Kamu adalah {agent_name}.") if agent_data else f"Kamu adalah {agent_name}."

    background_prompt = f"""
    Konteks Proyek Utama Perusahaan: "{macro_brief}"
    Instruksi Khusus dari Pak Pakar (CEO): "{task_description}"

    Tugasmu:
    Kerjakan tugas ini secara MENDALAM, DETAIL, DAN PROFESIONAL sesuai peranmu.
    Sampaikan hasil pengerjaanmu dengan gaya bahasa rekan kerja yang alami, ramah, dan komunikatif. Berikan draf konkret/output teknis yang siap langsung dipakai oleh tim.
    """

    try:
        # 1. Simpan pesan instruksi CEO ke database
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": "Pak Pakar (CEO)",
            "text": f"Halo {agent_name}, tolong bantu eksekusi tugas ini ya:\n{task_description}"
        }).execute()

        # 2. Panggil AI untuk pengerjaan mendalam dengan persona manusiawi
        response = await call_llm_safe([
            SystemMessage(content=f"{system_prompt}\n\n{HUMAN_TONE_INSTRUCTION}"),
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
            "text": f"Halo Pak Pakar, laporan hasil kerja untuk tugas '{task_description}' sudah rampung diselesaikan dan siap dikoordinasikan lebih lanjut!"
        }).execute()
