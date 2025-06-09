from fastapi import FastAPI, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from db_utils import create_table, ensure_pgvector_installed
from vector_utils import add_markdown_document, get_similar_chunks
from llm_utils import build_prompt, ask_llm

app = FastAPI()

ensure_pgvector_installed()
create_table()

@app.post("/add_document")
async def add_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        return JSONResponse(status_code=400, content={"error": "Only .md files allowed"})
    contents = await file.read()
    text = contents.decode("utf-8")
    add_markdown_document(text)
    return {"status": f"Added from {file.filename}"}

@app.post("/query_chunks")
def query_chunks_endpoint(question: str = Form(...), k: int = Form(5)):
    chunks = get_similar_chunks(question, k)
    return {"chunks": chunks}


@app.post("/query")
def query_endpoint(payload: dict = Body(...)):
    context = get_similar_chunks(payload["question"])
    prompt = build_prompt(context, payload["question"])
    answer = ask_llm(prompt)
    return {"answer": answer, "context": context}
