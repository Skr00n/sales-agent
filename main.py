# main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from rag_search import rag_search
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
        You are a Sales Intelligence Agent working with structured sales CSV data.

        The retrieved context below may include:
        - A list of customers
        - Customers with active sales
        - Customers with no sales activity
        - Latest sales transactions
        - Sales aggregated by customer

        -------------------------------------------------
        RETRIEVED SALES DATA:
        {rag_context}
        -------------------------------------------------

        Your responsibilities:
        1. Identify which customers currently have sales vs no sales
        2. Highlight customers with no sales or declining activity
        3. Summarize recent sales trends where available
        4. Flag any risks or missed opportunities
        5. Recommend concrete next actions for the sales team

        Keep responses clear, actionable, and grounded in the data provided.
        """,
        agent=sales_agent,
        expected_output="""
        - Customer insights grouped by sales status
        - Identified risks or gaps
        - Clear next-best action recommendations
        """
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
        placeholder="Which customers need follow-up based on recent sales data?"
    ),
    outputs=gr.Textbox(
        label="Sales Intelligence Agent Response",
        lines=12
    ),
    title="Sales Intelligence AI Agent",
    description="RAG-powered sales analysis using PostgreSQL + pgvector and CrewAI"
)

demo.launch()

