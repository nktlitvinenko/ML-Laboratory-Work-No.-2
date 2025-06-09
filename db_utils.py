from sqlalchemy import create_engine, text
from config import Config

engine = create_engine(Config.get("DATABASE_URL"))

def ensure_pgvector_installed():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

def create_table():
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding VECTOR(384)
        );
        """))
        conn.commit()
