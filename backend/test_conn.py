from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Ambil URI dari .env
DATABASE_URL = os.getenv("DATABASE_URL")

# Buat SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Test koneksi
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print("✅ Connection successful!")
        print("PostgreSQL version:", result.fetchone()[0])
except Exception as e:
    print("❌ Connection failed:", e)
