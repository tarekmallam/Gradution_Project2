import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL", "").strip()
key = os.getenv("SUPABASE_API_KEY", "").strip()

print(f"URL: {url}")
print(f"KEY: {key[:8]}... (len: {len(key)})")

try:
    supabase = create_client(url, key)
    print("✅ Supabase client created successfully!")
    # Try a simple query
    data = supabase.table('user').select('*').limit(1).execute()
    print("Sample query result:", data)
except Exception as e:
    print("❌ Supabase error:", e)
    import traceback; traceback.print_exc()
