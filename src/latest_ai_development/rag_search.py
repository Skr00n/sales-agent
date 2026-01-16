# rag_search.py
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


load_dotenv()

# -----------------------------
# Embeddings
# -----------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# -----------------------------
# Qdrant Connection
# -----------------------------
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

vector_db = QdrantVectorStore(
    client=qdrant_client,
    collection_name="sales_data",
    embedding=embedding_model
)

# -----------------------------
# RAG Search
# -----------------------------
def rag_search(query: str, k: int = 3) -> str:
    results = vector_db.similarity_search(query, k=k)

    if not results:
        return "No relevant historical data found."

    return "\n".join(doc.page_content for doc in results)
