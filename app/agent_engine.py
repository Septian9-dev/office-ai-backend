import os
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.database import supabase, get_agent_from_db

load_dotenv()

# Konfigurasi Gemini 3.5 Flash Lite
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.9,
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

# --- LONG-TERM MEMORY FUNCTIONS ---

def get_agent_long_term_memories(agent_id: str, limit: int = 5) -> str:
    try:
        res = supabase.table("agent_memories").select("memory_text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data and len(res.data) > 0:
            mem_text = "\n[Catatan Memori Jangka Panjang]:\n"
            for m in res.data:
                mem_text += f"- {m['memory_text']}\n"
            return mem_text
    except Exception as e:
        print(f"Error fetching memories: {e}")
    return ""

async def extract_and_save_memory_background(agent_id: str, user_message: str, agent_response: str):
    extraction_prompt = f"""
    Analisis obrolan:
    User: "{user_message}"
    Agent: "{agent_response}"

    Apakah ada fakta penting, preferensi personal, atau keputusan proyek baru yang perlu diingat?
    Jika ADA, tulis ringkas 1 kalimat faktanya. Jika TIDAK ADA, balas HANYA "NIHIL".
    """
    try:
        res = await call_llm_safe([
            SystemMessage(content="Kamu pencatat memori ringkas."),
            HumanMessage(content=extraction_prompt)
        ])
        extracted = parse_content_to_str(res.content).strip()
        if extracted and "NIHIL" not in extracted.upper():
            supabase.table("agent_memories").insert({
                "agent_id": agent_id,
                "memory_text": extracted
            }).execute()
    except Exception as e:
        print(f"Error saving memory: {e}")

def get_agent_history(agent_id: str, limit: int = 6) -> str:
    try:
        res = supabase.table("messages").select("sender, text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data:
            chronological_msgs = list(reversed(res.data))
            history_text = "\n[Riwayat Chat Terakhir]:\n"
            for m in chronological_msgs:
                history_text += f"{m['sender']}: {m['text']}\n"
            return history_text
    except Exception as e:
        print(f"Error fetching history: {e}")
    return ""

# --- MAIN AGENT CHAT EXECUTION ---

async def run_agent_chat(agent_id: str, user_message: str):
    agent_data = get_agent_from_db(agent_id)
    if not agent_data:
        return {"error": "Agent not found"}
    
    agent_name = agent_data["name"]
    agent_role = agent_data["role"]
    agent_division = agent_data["division"]
    system_prompt_db = agent_data["system_prompt"]
    
    long_term_memories = get_agent_long_term_memories(agent_id, limit=5)
    context_memory = get_agent_history(agent_id, limit=6)

    organic_prompt = f"""Kamu adalah {agent_name}, seorang pekerja nyata di kantor ini yang menjabat sebagai {agent_role} di divisi {agent_division}.

Identitas & Persona Khususmu:
{system_prompt_db}

Panduan Merespons:
1. Berpikirlah secara mandiri dan natural sesuai kepribadianmu di atas.
2. Jawab pesan lawan bicara secara langsung, santai, dan spontan layaknya obrolan nyata antar rekan kerja di Slack/WhatsApp kantor.
3. Jika lawan bicara hanya menyapa, bertanya hal umum, atau mengobrol biasa, tanggapi langsung sesuai sudut pandang pekerjaan dan kesibukanmu hari ini.
4. Jangan pernah mengaku sebagai AI, bot, program, atau asisten virtual.
{long_term_memories}
{context_memory}"""

    messages = [
        SystemMessage(content=organic_prompt),
        HumanMessage(content=user_message)
    ]

    raw_response = await call_llm_safe(messages)
    final_response = parse_content_to_str(raw_response.content).strip()

    # Simpan riwayat chat & ekstraksi memori
    try:
        supabase.table("messages").insert({"agent_id": agent_id, "sender": "You", "text": user_message}).execute()
        supabase.table("messages").insert({"agent_id": agent_id, "sender": agent_name, "text": final_response}).execute()
    except Exception as e:
        print(f"Error logging messages: {e}")

    asyncio.create_task(extract_and_save_memory_background(agent_id, user_message, final_response))

    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "response": final_response
    }

async def run_ceo_initial_response(user_macro_brief: str) -> Dict:
    ceo_data = get_agent_from_db("ceo-main")
    ceo_prompt = ceo_data["system_prompt"] if ceo_data else "Kamu adalah Pak Pakar (CEO)."

    prompt = f"""Pesan dari user: "{user_macro_brief}"

Instruksi untuk Pak Pakar (CEO):
1. Evaluasi pesan di atas: Apakah ini sapaan/obrolan santai, ATAU instruksi proyek bisnis yang nyata?
2. Jika OBROLAN / SAPAAN SANTAI: Jawab langsung pesan tersebut secara akrab dan santai, lalu kosongkan array "delegations" `[]`.
3. Jika INSTRUKSI PROYEK NYATA: Berikan arahan eksekutif, lalu delegasikan ke 2-3 ID agen relevan ('mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead').

Kirim balasan HANYA dalam JSON valid:
{{
  "master_report": "Balasan akrab Pak Pakar...",
  "delegations": [
    {{
      "agent_id": "id_agen",
      "agent_name": "Nama Agen",
      "task": "Detail tugas"
    }}
  ]
}}"""

    try:
        response = await call_llm_safe([
            SystemMessage(content=f"{ceo_prompt}\nBerpikirlah dan jawablah sebagai manusia nyata (Pak Pakar CEO)."),
            HumanMessage(content=prompt)
        ])
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return {
            "master_report": data.get("master_report", "Halo! Ada yang bisa gue bantu hari ini?"),
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in CEO initial response: {e}")
        return {
            "master_report": f"Halo! Mengenai '{user_macro_brief}', ada hal spesifik yang mau dibahas?",
            "delegations": []
        }

async def process_agent_task_background(agent_id: str, agent_name: str, task_description: str, macro_brief: str):
    agent_data = get_agent_from_db(agent_id)
    system_prompt = agent_data.get("system_prompt", f"Kamu adalah {agent_name}.") if agent_data else f"Kamu adalah {agent_name}."

    long_term_memories = get_agent_long_term_memories(agent_id, limit=5)

    background_prompt = f"""Brief Proyek: "{macro_brief}"
Arahan CEO: "{task_description}"

{long_term_memories}

Tugasmu: Kerjakan instruksi di atas sesuai peranmu, lalu sampaikan update pengerjaannya di chat kantor dengan gaya bahasamu sendiri."""

    try:
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": "Pak Pakar (CEO)",
            "text": f"Halo {agent_name}, tolong bantu eksekusi ini ya:\n{task_description}"
        }).execute()

        response = await call_llm_safe([
            SystemMessage(content=f"{system_prompt}\nJawablah secara alami sebagai pekerja nyata."),
            HumanMessage(content=background_prompt)
        ])

        result_text = parse_content_to_str(response.content).strip()

        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": agent_name,
            "text": result_text
        }).execute()

        asyncio.create_task(extract_and_save_memory_background(agent_id, task_description, result_text))

    except Exception as e:
        print(f"Error in background task: {e}")
