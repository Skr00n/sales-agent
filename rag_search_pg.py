# rag_search_pg.py

import os
import json
from sqlalchemy import create_engine, text
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

#embedder = OllamaEmbeddings(model="nomic-embed-text")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def rag_search(query: str, k: int = 3) -> str:
    query_embedding = embedder.embed_query(query)

    sql = text("""
    SELECT content, metadata
    FROM sales_embeddings
    ORDER BY embedding <-> (:query_embedding)::vector
    LIMIT :k
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"query_embedding": query_embedding, "k": k}
        ).fetchall()

    if not rows:
        return "No relevant data found."

    return "\n\n".join(
        f"[Result {i+1}]\n{row.content}\nMetadata: {json.dumps(row.metadata)}"
        for i, row in enumerate(rows)
    )
