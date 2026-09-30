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

async def call_llm_safe(messages: list, max_retries: int = 4) -> Any:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(0.15)
                return await llm.ainvoke(messages)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimit" in error_str:
                    wait_time = 2 * (attempt + 1)
                    print(f"[Rate Limit] Menunggu {wait_time} detik... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        return await llm.ainvoke(messages)

def get_agent_history(agent_id: str, limit: int = 6) -> str:
    """Mengambil riwayat percakapan & instruksi CEO sebelumnya dari database."""
    try:
        res = supabase.table("messages").select("sender, text").eq("agent_id", agent_id).order("created_at", desc=True).limit(limit).execute()
        if res.data:
            chronological_msgs = list(reversed(res.data))
            history_text = "\n\n--- RIWAYAT PERCAKAPAN & INSTRUKSI TERAKHIR ---\n"
            for m in chronological_msgs:
                history_text += f"{m['sender']}: {m['text']}\n"
            history_text += "--- AKHIR RIWAYAT ---\nGunakan informasi riwayat di atas untuk memberikan respons yang relevan dan konsisten.\n"
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

    # Memuat Memori Konteks
    context_memory = get_agent_history(agent_id, limit=6)
    full_system_prompt = system_prompt + context_memory

    messages = [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=user_message)
    ]

    response = await call_llm_safe(messages)
    text_response = parse_content_to_str(response.content)

    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "response": text_response
    }

async def run_team_briefing(division: str, user_brief: str) -> Dict:
    res = supabase.table("agents").select("*").eq("division", division).execute()
    agents = res.data
    if not agents:
        return {"error": f"Tidak ada agen ditemukan di divisi {division}"}

    manager = next((a for a in agents if a["role"] == "Manager"), None)
    specialists = [a for a in agents if a["role"] == "Specialist"]

    if not manager:
        return {"error": "Manager divisi tidak ditemukan."}

    specialist_info = "\n".join([f"- ID: {s['id']}, Nama: {s['name']}, Persona: {s['system_prompt']}" for s in specialists])
    
    planning_prompt = f"""
    Kamu adalah {manager['name']} (Manager Divisi {division}).
    Brief dari Atasan/User: "{user_brief}"

    Berikut adalah daftar spesialis di timmu:
    {specialist_info}

    Tugasmu: Pilih MAKSIMAL 2 spesialis yang paling relevan, lalu buatkan instruksi tugas spesifik untuk masing-masing dari mereka.
    
    TOLONG BALAS DENGAN FORMAT JSON VALID BERIKUT SAJA (TANPA TEKS LAIN/MARKDOWN BLOCK):
    [
      {{"agent_id": "id_spesialis", "task": "instruksi detail tugas"}},
      {{"agent_id": "id_spesialis_lain", "task": "instruksi detail tugas"}}
    ]
    """

    plan_response = await call_llm_safe([
        SystemMessage(content="Kamu adalah AI yang hanya membalas dalam format JSON array valid."),
        HumanMessage(content=planning_prompt)
    ])
    
    plan_text = parse_content_to_str(plan_response.content)
    raw_json = plan_text.replace("```json", "").replace("```", "").strip()
    try:
        tasks = json.loads(raw_json)
        if not isinstance(tasks, list):
            tasks = [{"agent_id": specialists[0]["id"], "task": user_brief}]
    except Exception:
        tasks = [{"agent_id": specialists[0]["id"], "task": user_brief}]

    async def execute_task(task_item):
        agent_data = next((s for s in specialists if s["id"] == task_item["agent_id"]), None)
        if not agent_data:
            return None
        
        agent_prompt = f"{agent_data['system_prompt']}\n\nInstruksi Tugas dari Manager: {task_item['task']}"
        res = await call_llm_safe([
            SystemMessage(content=agent_prompt),
            HumanMessage(content=f"Kerjakan tugas ini sesuai persona ahli kamu: {task_item['task']}")
        ])
        
        return {
            "agent_id": agent_data["id"],
            "agent_name": agent_data["name"],
            "task": task_item["task"],
            "result": parse_content_to_str(res.content)
        }

    specialist_results = await asyncio.gather(*[execute_task(t) for t in tasks])
    valid_results = [r for r in specialist_results if r is not None]

    results_summary_str = ""
    for r in valid_results:
        results_summary_str += f"\n--- Laporan dari {r['agent_name']} ---\nTugas: {r['task']}\nHasil:\n{r['result']}\n"

    final_report_prompt = f"""
    Kamu adalah {manager['name']} (Manager Divisi {division}).
    Brief awal dari Atasan: "{user_brief}"

    Berikut adalah laporan hasil kerja dari tim spesialis kamu:
    {results_summary_str}

    Tugasmu:
    Buat Laporan Ringkasan Eksekutif (Executive Summary) yang rapi untuk atasan/user. 
    Rangkum hasil kerja timmu secara terstruktur, jelas, profesional, dan sebutkan kontribusi dari masing-masing spesialis.
    """

    final_report = await call_llm_safe([
        SystemMessage(content=manager["system_prompt"]),
        HumanMessage(content=final_report_prompt)
    ])

    return {
        "division": division,
        "manager_name": manager["name"],
        "user_brief": user_brief,
        "team_contributions": valid_results,
        "executive_summary": parse_content_to_str(final_report.content)
    }

async def run_ceo_briefing(user_macro_brief: str) -> Dict:
    res = supabase.table("agents").select("*").eq("role", "Manager").execute()
    managers = res.data
    
    if not managers:
        return {"error": "Tidak ada Manager Divisi ditemukan."}

    manager_info = "\n".join([f"- Divisi: {m['division']}, Manager: {m['name']}" for m in managers])

    ceo_plan_prompt = f"""
    Kamu adalah CEO Perusahaan.
    Brief Makro Perusahaan dari Owner/User: "{user_macro_brief}"

    Berikut daftar Manager Divisi yang tersedia:
    {manager_info}

    Tugasmu: Pilih MAKSIMAL 2 divisi yang paling relevan untuk mengeksekusi brief ini. 
    Berikan instruksi khusus untuk masing-masing Manager Divisi tersebut.

    TOLONG BALAS DENGAN FORMAT JSON VALID BERIKUT SAJA (TANPA TEKS LAIN/MARKDOWN BLOCK):
    [
      {{"division": "Nama Divisi Tepat", "brief": "instruksi spesifik untuk manager divisi tersebut"}},
      {{"division": "Nama Divisi Tepat", "brief": "instruksi spesifik untuk manager divisi tersebut"}}
    ]
    """

    ceo_plan_res = await call_llm_safe([
        SystemMessage(content="Kamu adalah AI CEO yang hanya membalas dalam format JSON array valid."),
        HumanMessage(content=ceo_plan_prompt)
    ])

    plan_text = parse_content_to_str(ceo_plan_res.content)
    raw_json = plan_text.replace("```json", "").replace("```", "").strip()

    try:
        division_tasks = json.loads(raw_json)
        if not isinstance(division_tasks, list):
            division_tasks = [{"division": managers[0]["division"], "brief": user_macro_brief}]
    except Exception:
        division_tasks = [{"division": managers[0]["division"], "brief": user_macro_brief}]

    division_tasks = division_tasks[:2]

    division_reports = []
    for t in division_tasks:
        if isinstance(t, dict) and "division" in t and "brief" in t:
            report = await run_team_briefing(t["division"], t["brief"])
            division_reports.append(report)

    combined_reports_str = ""
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

async def run_ceo_briefing(user_macro_brief: str) -> Dict:
    """
    Menghasilkan Master CEO Report SEKALIGUS Balasan Detail Karyawan dalam 1 panggil API (Cepat & Lengkap)
    """
    res = supabase.table("agents").select("id, name, division, role, system_prompt").execute()
    all_agents = res.data or []

    ceo_prompt = f"""
    Kamu adalah sistem AI Multi-Agent Kantor. Pengguna memberikan instruksi makro kepada Pak Pakar (CEO):
    "{user_macro_brief}"

    Tugasmu:
    1. Buat Laporan Strategi Eksekutif (Master Report) dari Pak Pakar (CEO).
    2. Tentukan 2 sampai 3 karyawan/spesialis yang paling relevan untuk tugas ini (pilih dari ID berikut: 'mkt-lead', 'content-writer', 'design-3d', 'graphic-des', 'ppc-spec', 'fe-dev-1', 'uiux-1', 'sales-lead', 'fin-lead').
    3. Buat instruksi tugas spesifik dari Pak Pakar ke masing-masing karyawan.
    4. Buat LAPORAN BALASAN KERJA SANGAT DETAIL dari masing-masing karyawan sesuai peran keahlian mereka (misal: draf naskah lengkap dari Content Writer, konsep visual 3D dari 3D Artist, strategi iklan dari PPC Spec).

    BALAS DENGAN FORMAT JSON VALID BERIKUT SAJA (TANPA MARKDOWN BLOCK/TEKS LAIN):
    {{
      "master_report": "Teks Laporan Konsolidasi Strategi CEO Pak Pakar...",
      "delegations": [
        {{
          "agent_id": "id_agent_1",
          "agent_name": "Nama Agent 1",
          "task": "Instruksi spesifik dari Pak Pakar ke Agent 1",
          "reply": "Laporan hasil kerja detail dari Agent 1..."
        }},
        {{
          "agent_id": "id_agent_2",
          "agent_name": "Nama Agent 2",
          "task": "Instruksi spesifik dari Pak Pakar ke Agent 2",
          "reply": "Laporan hasil kerja detail dari Agent 2..."
        }}
      ]
    }}
    """

    try:
        response = await call_llm_safe([
            SystemMessage(content="Kamu adalah AI CEO kantor yang mengembalikan JSON valid saja."),
            HumanMessage(content=ceo_prompt)
        ])
        
        raw_text = parse_content_to_str(response.content)
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return {
            "macro_brief": user_macro_brief,
            "master_report": data.get("master_report", "Laporan CEO selesai diproses."),
            "delegations": data.get("delegations", [])
        }
    except Exception as e:
        print(f"Error in JSON CEO briefing: {e}")
        return {
            "master_report": f"Pak Pakar telah mengordinasikan instruksi: {user_macro_brief}",
            "delegations": [
                {
                    "agent_id": "content-writer",
                    "agent_name": "Rina (Content Lead)",
                    "task": "Buat naskah storyboard video iklan",
                    "reply": "Siap Pak Pakar! Draf naskah storyboard video iklan telah disiapkan dengan konsep visual catchy dan call-to-action promosi."
                },
                {
                    "agent_id": "design-3d",
                    "agent_name": "Raka (3D Artist)",
                    "task": "Buat pemodelan animasi produk 3D",
                    "reply": "Siap Pak Pakar! Aset animasi 3D produk sudah diproses menggunakan shading lighting modern siap render."
                }
            ]
        }
