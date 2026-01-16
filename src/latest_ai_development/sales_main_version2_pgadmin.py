# main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from .rag_search_version2 import rag_search

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = LLM(
    model="ollama/llama3.1",
    base_url="http://192.168.1.96:11434",
    temperature=0
)
#ollama url: 192.168.1.96:11434

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
    user_query = input("Ask a sales question: ").strip()

    rag_context = rag_search(user_query)

    task = Task(
        description=f"""
        You are a Sales Intelligence Agent.

        Relevant sales information retrieved from memory:
        -------------------------------------------------
        {rag_context}
        -------------------------------------------------

        Based on the above information:
        - Analyze the situation
        - Identify insights or risks
        - Provide clear recommendations
        """,
        agent=sales_agent,
        expected_output="Clear sales insights and recommendations"
    )

    crew = Crew(
        agents=[sales_agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()

    print("\n========== SALES AGENT OUTPUT ==========\n")
    print(result)
