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

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Easy Stones",
    embedding=embedding_model,
)

# -----------------------------
# RAG Search
# -----------------------------
def rag_search(query: str, owner: str | None = None, k: int = 3) -> str:
    """
    Semantic search over sales data.

    owner: optional dataset owner (e.g. 'alberts', 'prudhvis', 'jessica')
    """

    print(f"\n🔍 Query: {query}")
    if owner:
        print(f"👤 Owner filter: {owner}")

    search_kwargs = {}

    # 👇 THIS IS WHERE THE OWNER LOGIC LIVES
    if owner:
        search_kwargs["filter"] = {
            "must": [
                {
                    "key": "owner",
                    "match": {"value": owner.lower()}
                }
            ]
        }

    results = vector_db.similarity_search(
        query,
        k=k,
        **search_kwargs
    )

    print(f"📄 Results found: {len(results)}")

    if not results:
        return "No relevant historical data found."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "unknown")
        owner_meta = doc.metadata.get("owner", "unknown")
        formatted.append(
            f"[Result {i} | Owner: {owner_meta} | Source: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)
