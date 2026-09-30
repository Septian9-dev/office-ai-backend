import os
import json
import asyncio
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.database import supabase, get_agent_from_db

load_dotenv()

# Konfigurasi model Gemini 3.5 Flash Lite
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.85,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ATURAN GLOBAL REASONING & FORMAT BEBAS ROBOTIK
REASONING_AND_HUMAN_PROMPT = """
DILARANG KERAS MENGGUNAKAN BAHASA ROBOTIK (misal: "Tentu", "Sebagai AI", "Poin tersebut telah saya pahami", "Berikut adalah").

REASONING LOOP (ANALISIS INTERNAL SEBELUM MENJAWAB):
Sebelum memberikan balasan akhir, lakukan analisis internal singkat di dalam tag <thinking>...</thinking>:
<thinking>
1. Apa inti dari pesan pengguna dan apa konteks dari Long-Term Memory/riwayat obrolan yang relevan?
2. Bagaimana persona, gaya bicara, dan ciri khas spesifikku menyikapi hal ini secara alami & tidak kaku?
3. Apa tindakan atau jawaban paling tepat, relevan, dan manusiawi?
</thinking>

TULIS BALASAN AKHIR DILUAR TAG <thinking>. Balasan harus luwes, komunikatif, dan fleksibel sesuai persona agen.
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

def clean_thinking_process(text: str) -> str:
    """Menghapus blok <thinking>...</thinking> agar tidak muncul di UI pengguna."""
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return cleaned.strip()

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

def get_agent_long_term_memories(agent_id: str, limit: int = 10) -> str:
    """Mengambil fakta/ingatan jangka panjang yang pernah dicatat oleh agen."""
    try:
        res = supabase.table("agent_memories").select("memory_text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data and len(res.data) > 0:
            mem_text = "\n--- INGATAN JANGKA PANJANG (LONG-TERM MEMORY) ---\n"
            for m in res.data:
                mem_text += f"• {m['memory_text']}\n"
            mem_text += "--------------------------------------------------\n"
            return mem_text
    except Exception as e:
        print(f"Error fetching long term memories: {e}")
    return ""

async def extract_and_save_memory_background(agent_id: str, user_message: str, agent_response: str):
    """Mengekstrak informasi penting dari percakapan dan menyimpannya ke memori jangka panjang."""
    extraction_prompt = f"""
    Analisis percakapan berikut:
    User: "{user_message}"
    Agent: "{agent_response}"

    Apakah ada fakta penting, preferensi user, keputusan proyek, atau instruksi khusus yang perlu DIINGAT DALAM JANGKA PANJANG?
    Jika ADA, tuliskan 1-2 kalimat ringkas faktanya saja.
    Jika TIDAK ADA (hanya obrolan biasa/sapaan umum), balas HANYA dengan kata "NIHIL".
    """
    try:
        res = await call_llm_safe([
            SystemMessage(content="Kamu adalah pemproses memori AI."),
            HumanMessage(content=extraction_prompt)
        ])
        extracted = parse_content_to_str(res.content).strip()
        if extracted and "NIHIL" not in extracted.upper():
            supabase.table("agent_memories").insert({
                "agent_id": agent_id,
                "memory_text": extracted
            }).execute()
            print(f"[Memory Saved for {agent_id}]: {extracted}")
    except Exception as e:
        print(f"Error saving memory: {e}")

def get_agent_history(agent_id: str, limit: int = 6) -> str:
    try:
        res = supabase.table("messages").select("sender, text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data:
            chronological_msgs = list(reversed(res.data))
            history_text = "\n--- RIWAYAT CHAT TERAKHIR ---\n"
            for m in chronological_msgs:
                history_text += f"{m['sender']}: {m['text']}\n"
            history_text += "-----------------------------\n"
            return history_text
    except Exception as e:
        print(f"Error fetching history: {e}")
    return ""

# --- MAIN AGENT CHAT EXECUTION ---

async def run_agent_chat(agent_id: str, user_message: str):
    agent_data = get_agent_from_db(agent_id)
    if not agent_data:
        return {"error": "Agent not found"}
    
    system_prompt = agent_data["system_prompt"]
    agent_name = agent_data["name"]
    
    # 1. Ambil Memori Jangka Panjang & Riwayat Obrolan
    long_term_memories = get_agent_long_term_memories(agent_id, limit=8)
    context_memory = get_agent_history(agent_id, limit=6)

    # 2. Susun prompt gabungan yang memprioritaskan Persona SQL
    full_system_prompt = f"""=== PERSONA SPESIFIK & GAYA BICARA AGEN (DARI DATABASE) ===
{system_prompt}

=== ATURAN REASONING & FORMAT BAHASA ===
{REASONING_AND_HUMAN_PROMPT}

{long_term_memories}
{context_memory}"""

    messages = [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=user_message)
    ]

    # 3. Panggil LLM (Reasoning Loop)
    raw_response = await call_llm_safe(messages)
    full_text = parse_content_to_str(raw_response.content)
    
    # 4. Bersihkan pemikiran internal (<thinking>) untuk balasan pengguna
    final_response = clean_thinking_process(full_text)

    # 5. Jalankan ekstraksi memori jangka panjang secara async di background
    asyncio.create_task(extract_and_save_memory_background(agent_id, user_message, final_response))

    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "response": final_response
    }

async def run_ceo_initial_response(user_macro_brief: str) -> Dict:
    ceo_data = get_agent_from_db("ceo-main")
    ceo_prompt = ceo_data["system_prompt"] if ceo_data else "Kamu adalah Pak Pakar (CEO)."

    prompt = f"""
    Pengguna memberikan briefing makro:
    "{user_macro_brief}"

    Tugasmu:
    1. Lakukan analisis internal dulu di <thinking>...</thinking>.
    2. Jawab seperti Pak Pakar (CEO) sesuai persona dan gaya bicaramu.
    3. Pilih 2-3 ID agen relevan ('mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead').

    BALAS HANYA FORMAT JSON VALID INI (JANGAN MASUKKAN TAG THINKING KE DALAM JSON):
    {{
      "master_report": "Chat balasan Pak Pakar yang santai dan tegas ke user...",
      "delegations": [
        {{
          "agent_id": "id_agen_1",
          "agent_name": "Nama Agen 1",
          "task": "Instruksi dari Pak Pakar ke Agen 1"
        }}
      ]
    }}
    """

    try:
        response = await call_llm_safe([
            SystemMessage(content=f"{ceo_prompt}\n\n{REASONING_AND_HUMAN_PROMPT}"),
            HumanMessage(content=prompt)
        ])
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        master_report = clean_thinking_process(data.get("master_report", "Sip, instruksi udah diterusin ke tim ya!"))

        return {
            "master_report": master_report,
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in CEO initial response: {e}")
        return {
            "master_report": f"Sip, instruksi '{user_macro_brief}' udah gue terima. Tim terkait langsung jalan ya!",
            "delegations": [
                {"agent_id": "content-writer", "agent_name": "Rina (Content Lead)", "task": "Bikin draf skrip iklan."},
                {"agent_id": "design-3d", "agent_name": "Raka (3D Artist)", "task": "Siapin visual 3D-nya."}
            ]
        }

async def process_agent_task_background(agent_id: str, agent_name: str, task_description: str, macro_brief: str):
    agent_data = get_agent_from_db(agent_id)
    system_prompt = agent_data.get("system_prompt", f"Kamu adalah {agent_name}.") if agent_data else f"Kamu adalah {agent_name}."

    long_term_memories = get_agent_long_term_memories(agent_id, limit=5)

    background_prompt = f"""
    Proyek Utama: "{macro_brief}"
    Arahan Pak Pakar: "{task_description}"

    {long_term_memories}

    Tugasmu:
    Kerjakan tugas ini secara mendalam sesuai peranmu, lalu sampaikan balasan seperti kamu lagi kirim laporan di chat Slack kantor sesuai persona dan gaya bicaramu.
    """

    try:
        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": "Pak Pakar (CEO)",
            "text": f"Halo {agent_name}, tolong bantu eksekusi ini ya:\n{task_description}"
        }).execute()

        response = await call_llm_safe([
            SystemMessage(content=f"{system_prompt}\n\n{REASONING_AND_HUMAN_PROMPT}"),
            HumanMessage(content=background_prompt)
        ])

        raw_text = parse_content_to_str(response.content)
        result_text = clean_thinking_process(raw_text)

        supabase.table("messages").insert({
            "agent_id": agent_id,
            "sender": agent_name,
            "text": result_text
        }).execute()

        # Ekstrak fakta penting dari tugas ini ke Long-Term Memory
        asyncio.create_task(extract_and_save_memory_background(agent_id, task_description, result_text))

    except Exception as e:
        print(f"Error in background task: {e}")
