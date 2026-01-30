# rag_search.py
import os
from sqlalchemy import create_engine, text
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Database (Cloud SQL)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# -----------------------------
# Ollama Embeddings
# -----------------------------
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------
# RAG Search
# -----------------------------
def rag_search(query: str, k: int = 5) -> str:
    print(f"\n🔍 Query: {query}")

    query_embedding = embedding_model.embed_query(query)

    sql = text("""
        SELECT content
        FROM sales_embeddings
        ORDER BY embedding <-> :embedding
        LIMIT :k
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"embedding": query_embedding, "k": k}
        ).fetchall()

    if not rows:
        return "No relevant sales data found."

    return "\n\n".join(row[0] for row in rows)
