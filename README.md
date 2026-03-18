## Sales Intelligence Agent (`sales_agent`)

This project is a **Sales Intelligence Agent** built with **CrewAI**, **Gradio**, and **Postgres + pgvector**.  
It analyzes your sales data, runs RAG over invoices, and surfaces **actionable insights** on:

- **Customer activity** (active vs inactive)
- **Good vs bad debt**
- **Total / average sales & transaction volume**
- **Top / bottom customers and reps**
- **Next‑best actions and risks**

The main interactive experience is a web UI served from `main.py`, where the `sales_agent` answers questions grounded in your database and computed metrics.

---

## 1. Prerequisites

- **Python**: >= 3.10 and < 3.14
- **Postgres** with the **pgvector** extension enabled
- **Ollama** running locally (for embeddings and chat), or OpenAI if you switch models in code
- Recommended: a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`)

---

## 2. Environment configuration

Create a `.env` file in the project root with at least:

```bash
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB_NAME

# If you switch to OpenAI in code:
OPENAI_API_KEY=sk-...
```

If you are using **Ollama** (the default in `main.py` and `index_pgvector.py`), also ensure:

- Ollama is installed and running
- The `nomic-embed-text` and `llama3` models are pulled, for example:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

---

## 3. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

If you are using a different dependency workflow (e.g. `uv`), install the same Python packages declared for CrewAI, Gradio, SQLAlchemy, pgvector, and LangChain integrations.

---

## 4. Prepare the database

1. **Enable pgvector** in your Postgres database (run as a superuser):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. **Create the underlying sales tables** (for example `sales_analysis_2025`) and load your invoice / sales data.

3. **Create the embeddings table** used by the app (if not already present). A typical schema looks like:

```sql
CREATE TABLE IF NOT EXISTS sales_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR
);
```

Adjust column types to match how you configured pgvector in your environment.

---

## 5. Generate embeddings (`index_pgvector.py`)

The script `index_pgvector.py` connects to Postgres, reads rows from `sales_analysis_2025`, builds natural‑language descriptions, and stores **vector embeddings** into `sales_embeddings` using `nomic-embed-text` via Ollama.

Run it once (or whenever sales data changes significantly):

```bash
python index_pgvector.py
```

On success, you should see:

```text
✅ Cloud SQL data embedded into pgvector
```

These embeddings power the RAG search used by the `sales_agent`.

---

## 6. Run the Sales Agent UI (`main.py`)

The main Sales Intelligence Agent is defined and served from the root‑level `main.py`:

- `sales_agent` is a CrewAI `Agent` that uses the configured LLM.
- `rag_search_pg.py` performs vector search over `sales_embeddings`.
- `metrics.py` computes sales KPIs (totals, averages, rep performance, customer extremes, good/bad debt, etc.).
- A **Gradio** UI exposes KPIs and a chat box.

Start the app:

```bash
python main.py
```

By default, Gradio launches on:

- Host: `0.0.0.0`
- Port: `8080`

Open `http://localhost:8080` in your browser and you’ll see:

- A KPI row with **Total Sales**, **Transactions**, **Average Sale**, and **Top Rep**.
- A **Sales Agent Chatbot** textbox. Ask questions like:
  - “Which customers have the highest sales and are at risk?”
  - “Summarize our good vs bad debt.”
  - “How are my sales reps performing this month?”

The `sales_agent` will answer using **only** your RAG context and metrics.

---

## 7. Alternate CLI sales agent (`src/latest_ai_development/sales.py`)

There is also a CLI‑based Sales Intelligence Agent in `src/latest_ai_development/sales.py` that uses:

- OpenAI embeddings + Qdrant vector store
- CrewAI `Agent` + `Task` + `Crew`

To experiment with that version (after installing and configuring OpenAI and Qdrant):

```bash
python src/latest_ai_development/sales.py
```

You’ll be prompted for a sales rep ID and receive a text‑only analysis.

---

## 8. Development notes

- The original **LatestAiDevelopment Crew** template entrypoint (`src/latest_ai_development/main.py`) is still present but no longer the primary run mode.
- The **recommended path** to use this project is:
  1. Configure `.env` and Postgres.
  2. Run `python index_pgvector.py` to generate embeddings.
  3. Run `python main.py` and interact with the `sales_agent` in the browser.

This README describes how to run and extend the **Sales Intelligence Agent** rather than the generic CrewAI template.
