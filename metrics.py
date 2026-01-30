# metrics.py

import os
from datetime import date
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# -----------------------------
# Basic Metrics
# -----------------------------
def get_total_sales():
    sql = "SELECT COALESCE(SUM(invoice_amount),0) FROM sales_analysis_2025"
    with engine.connect() as c:
        return float(c.execute(text(sql)).scalar())

def get_average_sales():
    sql = "SELECT COALESCE(AVG(invoice_amount),0) FROM sales_analysis_2025"
    with engine.connect() as c:
        return float(c.execute(text(sql)).scalar())

def get_transactions_count():
    sql = "SELECT COUNT(*) FROM sales_analysis_2025"
    with engine.connect() as c:
        return int(c.execute(text(sql)).scalar())

# -----------------------------
# Sales by Rep
# -----------------------------
def get_sales_by_rep():
    sql = """
    SELECT sales_rep, SUM(invoice_amount) AS total_sales
    FROM sales_analysis_2025
    GROUP BY sales_rep
    ORDER BY total_sales DESC
    """
    with engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(text(sql))]

# -----------------------------
# Highest / Lowest Customers
# -----------------------------
def get_customer_extremes():
    sql = """
    SELECT customer, SUM(invoice_amount) AS total_sales
    FROM sales_analysis_2025
    GROUP BY customer
    ORDER BY total_sales DESC
    """
    with engine.connect() as c:
        rows = c.execute(text(sql)).fetchall()

    if not rows:
        return {}

    return {
        "highest": dict(rows[0]._mapping),
        "lowest": dict(rows[-1]._mapping)
    }

# -----------------------------
# Active vs Inactive Customers
# -----------------------------
def get_active_inactive_customers(days=30):
    sql = """
    SELECT customer, MAX(invoice_date) AS last_purchase,
           SUM(invoice_amount) AS total_spent
    FROM sales_analysis_2025
    GROUP BY customer
    """

    today = date.today()
    active, inactive = [], []

    with engine.connect() as c:
        rows = c.execute(text(sql)).fetchall()

    for r in rows:
        days_since = (today - r.last_purchase).days
        entry = {
            "customer": r.customer,
            "last_purchase": str(r.last_purchase),
            "total_spent": float(r.total_spent)
        }
        (active if days_since <= days else inactive).append(entry)

    return {
        "active_customers": active,
        "inactive_customers": inactive
    }

# -----------------------------
# Good vs Bad Debt
# -----------------------------
def get_good_bad_debt(days=60):
    sql = "SELECT invoice_amount, invoice_date FROM sales_analysis_2025"
    today = date.today()
    good, bad = 0, 0

    with engine.connect() as c:
        rows = c.execute(text(sql)).fetchall()

    for r in rows:
        age = (today - r.invoice_date).days
        if age <= days:
            good += r.invoice_amount
        else:
            bad += r.invoice_amount

    return {
        "good_debt": round(good, 2),
        "bad_debt": round(bad, 2)
    }
