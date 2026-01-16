from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pathlib import Path
import csv

# ------------------------------------------------
# Environment
# ------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

#correct path for csv file
CSV_FILE1 = BASE_DIR / "sales_data.csv"
CSV_FILE2 = BASE_DIR / "customer_data.csv"
# ------------------------------------------------
# Embedding Model
# ------------------------------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# ------------------------------------------------
# Qdrant Vector Store (auto-create collection)
# ------------------------------------------------
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

vector_db = QdrantVectorStore(
    client=qdrant_client,
    collection_name="sales_data",
    embedding=embedding_model
)

#converts rows to natural language text
def csv_to_documents():
    docs = []
    
    with open(CSV_FILE1, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            docs.append(
                f"Sales Rep {row['rep_id']} manages {row['customers']} customers. "
                f"Monthly revenue is ${row['monthly_revenue']}. "
                f"Target is ${row['target']}. "
                f"Conversion rate is {row['conversion_rate']}."
            )
    with open(CSV_FILE2, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(
                f"Customer {row['customer_id']} has "
                f"average monthly sales of ${row['avg_monthly_sales']}."
            )

    return docs
# ------------------------------------------------
def load_sales_data():
    sales_reps = []
    customers = []

    with open(CSV_FILE1, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            sales_reps.append({
                "rep_id": row["rep_id"].strip(),
                "customers": int(row["customers"]),
                "monthly_revenue": float(row["monthly_revenue"]),
                "target": float(row["target"]),
                "conversion_rate": float(row["conversion_rate"])
            })

    with open(CSV_FILE2, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            customers.append({
                "customer_id": row["customer_id"].strip(),
                "avg_monthly_sales": float(row["avg_monthly_sales"])
            })

    return {
        "sales_reps": sales_reps,
        "customers": customers
    }

if __name__ == "__main__":
    docs = csv_to_documents()
    vector_db.add_texts(docs)
    print("csv data vectorized and stored in Qdrant")