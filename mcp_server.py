from fastmcp import FastMCP
from mcp.server.auth.providers.bearer import BearerAuthProvider
from pydantic import Field
from typing import Annotated
from flask import Flask, request
import sqlite3, pandas as pd
from sarvamai import SarvamAI
import os

app = Flask(__name__)
mcp = FastMCP(server_name="nl_sql_tool")

# Auth setup
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # same token used in Flask
mcp.set_auth_provider(BearerAuthProvider(lambda token: token == AUTH_TOKEN))

client = SarvamAI(api_subscription_key=os.getenv("SARVAMAI_KEY"))
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")

@mcp.tool(description="Upload CSV from URL and store in database")
async def upload_csv_url(
    file_url: Annotated[str, Field(description="Publicly accessible CSV URL")]
) -> str:
    df = pd.read_csv(file_url)
    conn = sqlite3.connect(DATABASE_PATH)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()
    return "CSV uploaded successfully."

@mcp.tool(description="Ask a natural language question about the data")
async def ask_data_question(
    question: Annotated[str, Field(description="Natural language question")]
) -> str:
    prompt = f"You are an assistant that converts natural language questions into SQL for table 'data'.\nQuestion: {question}\nOnly return the SQL."
    resp = client.chat.completions(messages=[{"role": "user", "content": prompt}], max_tokens=100, temperature=0)
    sql = resp.choices[0].message.content.strip()
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df.to_json(orient="records")
