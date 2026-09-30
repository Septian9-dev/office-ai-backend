import os
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.database import supabase, get_agent_from_db

load_dotenv()

# Menggunakan model Gemini 3.5 Flash Lite
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def parse_content_to_str(content: Any) -> str:
    """Helper untuk mengonversi respon content LLM (baik string maupun list) menjadi string murni."""
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

# Semaphore & Retry Manager untuk mencegah error 429 (Rate Limit 15 RPM Free Tier)
semaphore = asyncio.Semaphore(2)  # Maksimal 2 request diproses bersamaan

async def call_llm_safe(messages: list, max_retries: int = 4) -> Any:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Jeda 0.8 detik antar request agar tidak melebihi batas 15 RPM
                await asyncio.sleep(0.8)
                return await llm.ainvoke(messages)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimit" in error_str:
                    wait_time = 4 * (attempt + 1)
                    print(f"[Rate Limit] Kuota tercapai, menunggu {wait_time} detik... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        return await llm.ainvoke(messages)

async def run_agent_chat(agent_id: str, user_message: str):
    agent_data = get_agent_from_db(agent_id)
    if not agent_data:
        return {"error": "Agent not found"}
    
    system_prompt = agent_data["system_prompt"]
    agent_name = agent_data["name"]

    messages = [
        SystemMessage(content=system_prompt),
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
    # 1. Ambil seluruh agen di divisi dari Supabase
    res = supabase.table("agents").select("*").eq("division", division).execute()
    agents = res.data
    if not agents:
        return {"error": f"Tidak ada agen ditemukan di divisi {division}"}

    manager = next((a for a in agents if a["role"] == "Manager"), None)
    specialists = [a for a in agents if a["role"] == "Specialist"]

    if not manager:
        return {"error": "Manager divisi tidak ditemukan."}

    # 2. Manager Menganalisis Brief & Membagi Tugas ke Maksimal 2 Spesialis
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

    # 3. Masing-Masing Spesialis Mengerjakan Sub-Tugasnya
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

    # 4. Manager Merangkum Seluruh Hasil Kerja Tim
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
    # 1. Ambil seluruh Manager Divisi dari database
    res = supabase.table("agents").select("*").eq("role", "Manager").execute()
    managers = res.data
    
    if not managers:
        return {"error": "Tidak ada Manager Divisi ditemukan."}

    manager_info = "\n".join([f"- Divisi: {m['division']}, Manager: {m['name']}" for m in managers])

    # 2. CEO memilih MAKSIMAL 2 divisi paling relevan (agar hemat kuota API)
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

    # Pembatasan keras maksimal 2 divisi untuk Free Tier API
    division_tasks = division_tasks[:2]

    # 3. Jalankan Briefing Divisi secara Teratur (Sequential Throttle)
    division_reports = []
    for t in division_tasks:
        if isinstance(t, dict) and "division" in t and "brief" in t:
            report = await run_team_briefing(t["division"], t["brief"])
            division_reports.append(report)

    # 4. CEO Merangkum Master Strategy Report
    combined_reports_str = ""
    for r in division_reports:
        if isinstance(r, dict) and "executive_summary" in r:
            combined_reports_str += f"\n=== LAPORAN DIVISI {r['division'].upper()} ({r['manager_name']}) ===\n"
            combined_reports_str += f"Brief Divisi: {r['user_brief']}\n"
            combined_reports_str += f"Hasil Exec Summary:\n{r['executive_summary']}\n"

    final_ceo_prompt = f"""
    Kamu adalah Pak Pakar (CEO).
    Brief Awal dari Owner/User: "{user_macro_brief}"

    Berikut adalah laporan konsolidasi dari para Manager Divisi kamu:
    {combined_reports_str}

    Tugasmu:
    Buat Master Executive Roadmap & Consolidated Strategy Report untuk Atasan/User.
    Sajikan dalam format profesional, terstruktur, serta berikan rekomendasi keputusan strategis CEO di bagian akhir.
    """

    final_ceo_report = await call_llm_safe([
        SystemMessage(content="Kamu adalah CEO perusahaan yang memberikan laporan level C-Suite."),
        HumanMessage(content=final_ceo_prompt)
    ])

    return {
        "macro_brief": user_macro_brief,
        "divisions_involved": [t["division"] for t in division_tasks if isinstance(t, dict) and "division" in t],
        "division_details": division_reports,
        "master_report": parse_content_to_str(final_ceo_report.content)
    }