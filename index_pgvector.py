# index_pgvector.py

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

SOURCE_SQL = """
SELECT invoice_id, customer, sales_rep, invoice_date, invoice_amount
FROM sales_analysis_2025
"""

with engine.begin() as conn:
    rows = conn.execute(text(SOURCE_SQL)).fetchall()

    for row in rows:
        content = (
            f"Invoice {row.invoice_id} for customer {row.customer} "
            f"handled by {row.sales_rep} on {row.invoice_date}. "
            f"Invoice amount was ${row.invoice_amount}."
        )

        embedding = embedder.embed_query(content)

        conn.execute(
            text("""
            INSERT INTO sales_embeddings (content, metadata, embedding)
            VALUES (:content, :metadata, :embedding)
            """),
            {
                "content": content,
                "metadata": json.dumps({
                    "invoice_id": row.invoice_id,
                    "customer": row.customer,
                    "sales_rep": row.sales_rep
                }),
                "embedding": embedding
            }
        )

print("✅ Cloud SQL data embedded into pgvector")
