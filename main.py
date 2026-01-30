# main.py

import os
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from rag_search_pg import rag_search
from metrics import (
    get_total_sales,
    get_average_sales,
    get_transactions_count,
    get_sales_by_rep,
    get_customer_extremes,
    get_active_inactive_customers,
    get_good_bad_debt
)

load_dotenv()
gr.close_all()

# -----------------------------
# LLM (Ollama)
# -----------------------------
#llm = LLM(
#    model="ollama/llama3",
#    temperature=0.2,
#    base_url="http://localhost:11434"
#)
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2
)

sales_agent = Agent(
    role="Sales Intelligence Agent",
    goal="Analyze sales performance and recommend next-best actions",
    backstory="Senior sales strategist with strong analytics expertise",
    llm=llm,
    verbose=True
)

# -----------------------------
# Core AI Function
# -----------------------------
def sales_agent_chat(user_query: str) -> str:
    rag_context = rag_search(user_query)

    metrics_block = f"""
TOTAL SALES: {get_total_sales()}
AVERAGE SALES: {get_average_sales()}
TOTAL TRANSACTIONS: {get_transactions_count()}

SALES BY REP:
{get_sales_by_rep()}

CUSTOMER EXTREMES:
{get_customer_extremes()}

CUSTOMER ACTIVITY:
{get_active_inactive_customers()}

GOOD VS BAD DEBT:
{get_good_bad_debt()}
"""

    task = Task(
        description=f"""
You are a Sales Intelligence Agent.

Use ONLY the data below.

---------------- RAG CONTEXT ----------------
{rag_context}

---------------- METRICS ----------------
{metrics_block}

Provide a FINAL business-ready answer including:
- Active vs inactive customers
- Good vs bad debt
- Total & average sales
- Highest & lowest customers
- Risks and next actions

IMPORTANT:
- Do NOT explain your thinking
- Do NOT say "I can"
- Output ONLY the final answer
""",
        agent=sales_agent,
        expected_output="Final structured business summary with insights and next actions."
    )

    crew = Crew(
        agents=[sales_agent],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return str(result)

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .kpi {
        padding: 20px;
        border-radius: 14px;
        background: #1e1e1e;
        text-align: center;
        font-size: 18px;
    }
    """
) as demo:

    gr.Markdown("## 📊 Sales Intelligence AI")

    with gr.Row():
        total_sales_md = gr.Markdown(elem_classes="kpi")
        transactions_md = gr.Markdown(elem_classes="kpi")
        avg_sales_md = gr.Markdown(elem_classes="kpi")
        top_rep_md = gr.Markdown(elem_classes="kpi")

    def load_kpis():
        reps = get_sales_by_rep()
        return (
            f"### 💰 Total Sales\n${get_total_sales():,.2f}",
            f"### 🔁 Transactions\n{get_transactions_count()}",
            f"### 📈 Average Sale\n${get_average_sales():,.2f}",
            f"### 🏆 Highest Rep\n{reps[0]['sales_rep'] if reps else 'N/A'}"
        )

    demo.load(
        load_kpis,
        outputs=[total_sales_md, transactions_md, avg_sales_md, top_rep_md]
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="Sales Agent Chatbot",
                placeholder="Ask about customers, debt, performance..."
            )
            submit_btn = gr.Button("Ask AI", variant="primary")

        with gr.Column(scale=2):
            output_box = gr.Markdown(label="Insights")

    submit_btn.click(
        sales_agent_chat,
        inputs=user_input,
        outputs=output_box
    )

#demo.launch(
#    server_name="0.0.0.0",
#    server_port=int(os.environ.get("PORT", 7860))
#)
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 8080)),
    share=False,
    show_error=True,
    quiet=True
)