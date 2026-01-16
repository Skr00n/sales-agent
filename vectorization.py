from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
#CSV Files
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "EasyStones"

CSV_FILES = list(DATA_DIR.rglob("*.csv"))

all_documents = []

#Load all CSVs
for csv_file in CSV_FILES:
    loader = CSVLoader(file_path=str(csv_file))
    docs = loader.load()

    for doc in docs:
        doc.metadata = {
            "source_file": csv_file.name,
            "source_path": str(csv_file),
        }

    print(f"Loaded {len(docs)} rows from {csv_file.name}")
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
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

#Qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="Easy Stones"
)

print("✅ Indexing complete for users, roles, and entitlements")
