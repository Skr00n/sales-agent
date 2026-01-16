# main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from .rag_search_version2 import rag_search
import gradio as gr

load_dotenv()

# -----------------------------
# LLM
# -----------------------------
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2
)

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
def sales_agent_chat(user_query:str) -> str:
    # if not user_query.strip():
    #     return "Ask a sales question: "

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
    #print(result)
    return str(result)
demo = gr.Interface(
    fn = sales_agent_chat,
    inputs=gr.Textbox(
        label="Ask a sales question: ",
        placeholder="Analyze REP_1's performance and suggest improvements"
    ),
    outputs=gr.Textbox(
        label="Sales Intelligence Agent Response",
        lines=12
    ),
    title="Sales Intelligence AI Agent",
    description="RAG-powered sales analysis using PostgreSQL + pgvector and CrewAI"
)

demo.launch()

def tell_joke(self) -> Task:
    return Task(
        config=self.task_config["tell_joke"],
        human_input=True,
    )