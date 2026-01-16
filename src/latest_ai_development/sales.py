"""
CrewAI Sales Agent using OpenAI + Qdrant (RAG)
---------------------------------------------
Capabilities:
1. Sales rep capacity by customer
2. Sales rep tracking
3. Sales targets by customer
4. Sales advisory (next best actions)
"""

from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import json

# ------------------------------------------------
# Environment
# ------------------------------------------------
load_dotenv()

# ------------------------------------------------
# Embedding Model
# ------------------------------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# ------------------------------------------------
# Qdrant Vector Store (auto-create collection)
# ------------------------------------------------
vector_db = QdrantVectorStore.from_documents(
    documents=[],  # start empty
    url="http://localhost:6333",
    collection_name="sales_data",
    embedding=embedding_model
)

# ------------------------------------------------
# LLM for CrewAI
# ------------------------------------------------
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2
)

# ------------------------------------------------
# Stub Sales Data (Replace with DB later)
# ------------------------------------------------
sales_data = {
    "sales_reps": [
        {
            "rep_id": "REP_1",
            "customers": 12,
            "monthly_revenue": 180000,
            "target": 200000,
            "conversion_rate": 0.21
        },
        {
            "rep_id": "REP_2",
            "customers": 6,
            "monthly_revenue": 95000,
            "target": 120000,
            "conversion_rate": 0.34
        }
    ],
    "customers": [
        {"customer_id": "CUST_1", "avg_monthly_sales": 40000},
        {"customer_id": "CUST_2", "avg_monthly_sales": 25000}
    ]
}

# ------------------------------------------------
# Convert Sales Data → Vectorizable Text
# ------------------------------------------------
def sales_data_to_documents(sales_data):
    documents = []

    for rep in sales_data["sales_reps"]:
        text = (
            f"Sales Rep {rep['rep_id']} manages {rep['customers']} customers. "
            f"Monthly revenue is ${rep['monthly_revenue']}. "
            f"Target is ${rep['target']}. "
            f"Conversion rate is {rep['conversion_rate']}."
        )
        documents.append(text)

    for cust in sales_data["customers"]:
        text = (
            f"Customer {cust['customer_id']} has "
            f"average monthly sales of ${cust['avg_monthly_sales']}."
        )
        documents.append(text)

    return documents

# ------------------------------------------------
# Store Sales Data in Qdrant (Run Once)
# ------------------------------------------------
docs = sales_data_to_documents(sales_data)

if docs:
    vector_db.add_texts(docs)
    print("✅ Sales data successfully vectorized and stored in Qdrant")

# ------------------------------------------------
# Helper: Get Sales Rep Data
# ------------------------------------------------
def get_rep_data(rep_id, sales_data):
    rep = next(
        (r for r in sales_data["sales_reps"] if r["rep_id"] == rep_id),
        None
    )

    if not rep:
        raise ValueError(f"Sales Rep ID '{rep_id}' not found.")

    return {
        "sales_rep": rep,
        "customers": sales_data["customers"]
    }

# ------------------------------------------------
# Sales Intelligence Agent
# ------------------------------------------------
sales_agent = Agent(
    role="Sales Intelligence Agent",
    goal="""
    Analyze sales rep performance, capacity, targets,
    and provide data-driven sales recommendations
    and next-best actions.
    """,
    backstory="""
    You are a senior sales strategist with deep experience
    in revenue growth, account planning, and sales analytics.
    """,
    llm=llm,
    verbose=True
)

# ------------------------------------------------
# Run Application
# ------------------------------------------------
if __name__ == "__main__":
    try:
        rep_id_input = input("Enter Sales Rep ID (e.g. REP_1): ").strip()

        # Live rep data
        rep_context = get_rep_data(rep_id_input, sales_data)

        # Retrieve relevant memory from Qdrant
        retrieved_docs = vector_db.similarity_search(
            f"Sales performance for {rep_id_input}",
            k=3
        )

        retrieved_context = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Create Task with RAG context
        sales_task = Task(
            description=f"""
            You are a Sales Intelligence Agent.

            Relevant historical sales data retrieved from memory:
            -----------------------------------------------------
            {retrieved_context}
            -----------------------------------------------------

            Current Sales Rep data:
            {json.dumps(rep_context, indent=2)}

            Perform the following:
            1. Evaluate capacity
            2. Analyze performance vs target
            3. Recommend realistic targets
            4. Provide next-best-action advice
            """,
            agent=sales_agent,
            expected_output="Sales insights for the selected rep"
        )

        crew = Crew(
            agents=[sales_agent],
            tasks=[sales_task],
            verbose=True
        )

        result = crew.kickoff()

        print("\n========== SALES AGENT OUTPUT ==========\n")
        print(result)

    except ValueError as e:
        print(f"Error: {e}")
