from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from db_utils import engine
import re

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

EMBEDDING_INSERT_TMPL = "INSERT INTO documents (content, embedding) VALUES (:c, :e)"
SIMILAR_CHUNKS_SELECT_TMPL = "SELECT content FROM documents ORDER BY embedding <-> (:q)::vector LIMIT :k;"

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_markdown(md_text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r'[`#*>\[\]\(\)\-!]', '', md_text)
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

def add_markdown_document(md_text: str):
    chunks = chunk_markdown(md_text)
    vectors = embed(chunks)
    with engine.connect() as conn:
        for chunk, vec in zip(chunks, vectors):
            conn.execute(text(EMBEDDING_INSERT_TMPL), {"c": chunk, "e": vec})
        conn.commit()

def get_similar_chunks(query: str, k: int = 5) -> list[str]:
    query_vec = embed([query])[0]
    with engine.connect() as conn:
        res = conn.execute(text(SIMILAR_CHUNKS_SELECT_TMPL), {"q": query_vec, "k": k}).fetchall()
        return [r[0] for r in res]
