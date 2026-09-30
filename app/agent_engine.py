import os
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.database import supabase, get_agent_from_db

load_dotenv()

# High Temperature (0.85) agar bahasa bervariasi dan tidak kaku
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.85,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# PROMPT KHUSUS: FEW-SHOT EXAMPLES + BANNED WORDS
FEW_SHOT_HUMAN_PROMPT = """
DILARANG KERAS MENGGUNAKAN KATA-KATA ROBOTIK BERIKUT:
❌ "Tentu,"
❌ "Sebagai [peran]..."
❌ "Poin tersebut telah saya pahami..."
❌ "Berikut adalah..."
❌ "Demikian laporan..."
❌ "Saya siap membantu Anda..."

GAYA BICARA KANTOR MODERN (MANUSIAWI & NATURAL):
- Bicara langsung seperti kirim chat di Slack/WhatsApp kantor.
- Pakai variasi kata: "Okee", "Sip", "Gini mas/mbak", "Aman", "Gass", "Btw", "Siap Pak".
- Boleh pakai singkatan wajar (yg, dkk, bgt, tetep).

CONTOH DIALOG MANUSIAWI (JADIKAN PATOKAN GAYA BICARA):

[Contoh Chat Pak Pakar - CEO]
User: "Pak Pakar, kita mau buat campaign baru untuk akhir tahun."
Pak Pakar: "Sip, mantap! Ide bagus tuh. Eko coba siapin konsep utamanya dulu ya, Raka minta tolong visual 3D-nya disiapin dari sekarang. Pokoknya akhir minggu ini gue mau liat draf awalnya ya team."

[Contoh Chat Rina - Content Lead]
Pak Pakar: "Rina, tolong buat skrip video iklan ya."
Rina: "Okee Pak Pakar! Ini draf kasar skripnya udah gue susun. Konsepnya dibuat rada pop & eye-catching di 5 detik pertama biar orang ga skip. Coba cek deh Pak:"

[Contoh Chat Raka - 3D Artist]
Pak Pakar: "Raka, animasi 3D produk gimana?"
Raka: "Aman Pak! Aset 3D-nya udah beres di-render. Lighting sama shading-nya udah gue bikin modern banget biar makin dapet feel premium-nya. Tinggal gabungin sama tim video."
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
            history_text = "\n\n--- RIWAYAT CHAT ---\n"
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

    full_system_prompt = f"{system_prompt}\n\n{FEW_SHOT_HUMAN_PROMPT}\n{context_memory}"

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
    prompt = f"""
    Kamu adalah Pak Pakar (CEO kantor). User kasih briefing:
    "{user_macro_brief}"

    Tugasmu:
    1. Jawab seperti CEO nyata di chat kantor (singkat, tegas, komunikatif, tanpa bahasa baku formal/robotik).
    2. Pilih 2-3 ID agen relevan dari: 'mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead'.
    3. Kasih arahan santai ke agen tersebut.

    BALAS HANYA FORMAT JSON VALID INI:
    {{
      "master_report": "Chat balasan Pak Pakar yang santai dan tegas ke user...",
      "delegations": [
        {{
          "agent_id": "id_agen_1",
          "agent_name": "Nama Agen 1",
          "task": "Instruksi santai dari Pak Pakar ke Agen 1"
        }}
      ]
    }}
    """

    try:
        response = await call_llm_safe([
            SystemMessage(content=f"Kamu adalah Pak Pakar, CEO startup yang santai dan luwes.\n{FEW_SHOT_HUMAN_PROMPT}"),
            HumanMessage(content=prompt)
        ])
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return {
            "master_report": data.get("master_report", "Sip, instruksi udah gue terusin ke tim ya!"),
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in CEO initial response: {e}")
        return {
            "master_report": f"Sip, ide '{user_macro_brief}' udah dikoordinasin. Tim langsung jalan ya!",
            "delegations": [
                {"agent_id": "content-writer", "agent_name": "Rina (Content Lead)", "task": "Bikin draf skrip iklan."},
                {"agent_id": "design-3d", "agent_name": "Raka (3D Artist)", "task": "Siapin visual 3D-nya."}
            ]
        }

async def process_agent_task_background(agent_id: str, agent_name: str, task_description: str, macro_brief: str):
    agent_data = get_agent_from_db(agent_id)
    system_prompt = agent_data.get("system_prompt", f"Kamu adalah {agent_name}.") if agent_data else f"Kamu adalah {agent_name}."

    background_prompt = f"""
    Proyek: "{macro_brief}"
    Arahan Pak Pakar: "{task_description}"

    Tugasmu:
    Kerjakan tugas ini dengan lengkap dan profesional, TAPI sampaikan seperti kamu lagi laporan di chat WhatsApp/Slack tim. Gaya bahasa santai, lugas, dan ga kaku sama sekali.
    """

    try:
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": "Pak Pakar (CEO)",
            "text": f"Halo {agent_name}, tolong bantu eksekusi ini ya:\n{task_description}"
        }).execute()

        response = await call_llm_safe([
            SystemMessage(content=f"{system_prompt}\n\n{FEW_SHOT_HUMAN_PROMPT}"),
            HumanMessage(content=background_prompt)
        ])

        result_text = parse_content_to_str(response.content)

        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": agent_name,
            "text": result_text
        }).execute()

    except Exception as e:
        print(f"Error in background task: {e}")
