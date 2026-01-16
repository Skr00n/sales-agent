# main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import json
from .rag_search import rag_search
from .vectorize_sales import load_sales_data

load_dotenv()
sales_data = load_sales_data()


# -----------------------------
# LLM
# -----------------------------
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2
)

# -----------------------------
# Helper: Get Rep Data
# -----------------------------
def get_rep_data(rep_id: str):
    rep = next(
        (r for r in sales_data["sales_reps"] if r["rep_id"] == rep_id),
        None
    )

    if not rep:
        raise ValueError(f"Sales Rep ID '{rep_id}' not found")

    return {
        "sales_rep": rep,
        "customers": sales_data["customers"]
    }

# -----------------------------
# Sales Intelligence Agent
# -----------------------------
sales_agent = Agent(
    role="Sales Intelligence Agent",
    goal="Analyze sales performance and recommend next-best actions",
    backstory="Senior sales strategist with strong analytics expertise",
    llm=llm,
    verbose=True
)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    try:
        rep_id = input("Enter Sales Rep ID (e.g. REP_1): ").strip()

        rep_context = get_rep_data(rep_id)

        rag_context = rag_search(
            f"Sales performance and targets for {rep_id}"
        )

        task = Task(
            description=f"""
            You are a Sales Intelligence Agent.

            Historical sales context (retrieved from vector DB):
            ----------------------------------------------------
            {rag_context}
            ----------------------------------------------------

            Current Sales Rep data:
            {json.dumps(rep_context, indent=2)}

            Perform:
            1. Capacity evaluation
            2. Performance vs target analysis
            3. Realistic target recommendation
            4. Next-best-action guidance
            """,
            agent=sales_agent,
            expected_output="Actionable sales intelligence insights"
        )

        crew = Crew(
            agents=[sales_agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()

        print("\n========== SALES AGENT OUTPUT ==========\n")
        print(result)

    except ValueError as e:
        print(f"Error: {e}")
