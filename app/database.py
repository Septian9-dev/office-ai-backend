import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_agent_from_db(agent_id: str):
    response = supabase.table("agents").select("*").eq("id", agent_id).execute()
    if response.data:
        return response.data[0]
    return None