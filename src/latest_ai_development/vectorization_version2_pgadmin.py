from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
import os

load_dotenv()
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+psycopg2://postgres:password@192.168.1.96:5432/postgres"
)
COLLECTION_NAME = "sales_embeddings"

BASE_DIR = Path(__file__).resolve().parent
#CSV Files
CSV_FILES = [
    BASE_DIR / "customer_data.csv",
    BASE_DIR / "sales_data.csv"
]

all_documents = []

#Load all CSVs
for csv_file in CSV_FILES:
    loader = CSVLoader(file_path=csv_file)
    docs = loader.load()
    print(f"Loaded {len(docs)} rows from {csv_file}")
    all_documents.extend(docs)
    
print(f"\nTotal documents loaded: {len(all_documents)}")

#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_documents)
print(f"Total chunks after splitting: {len(chunks)}")

#Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://192.168.1.96:11434"
)

#Qdrant

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True
)

print("✅ Indexing complete for users, roles, and entitlements")
