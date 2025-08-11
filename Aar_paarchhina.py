import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
import sqlite3
import pandas as pd
import json
from datetime import datetime

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("PUCH_AUTH_TOKEN")
MY_NUMBER = os.environ.get("PHONE_NUMBER")
DATABASE_PATH = os.environ.get("DATABASE_PATH", "./demo.db")
SARVAMAI_KEY = os.environ.get("SARVAMAI_KEY")

assert TOKEN is not None, "Please set PUCH_AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set PHONE_NUMBER in your .env file"

# Initialize Sarvam AI client if key is available
if SARVAMAI_KEY:
    from sarvamai import SarvamAI
    sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_KEY)
else:
    sarvam_client = None

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

def nl_to_sql_sarvam(nl_question: str, max_tokens: int = 100) -> str:
    """Convert natural language to SQL using Sarvam AI"""
    if not sarvam_client:
        raise ValueError("Sarvam AI client not initialized - check SARVAMAI_KEY")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("PRAGMA table_info(data)")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()

    prompt = (
        f"You are an assistant that converts natural language questions into SQL queries for a SQLite table named 'data'.\n"
        f"Available columns: {', '.join(columns)}\n"
        f"Question: {nl_question}\n"
        f"Only return the SQL query, nothing else."
    )
    
    try:
        resp = sarvam_client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        sql_query = resp.choices[0].message.content.strip()
        return sql_query
    except Exception as e:
        raise ValueError(f"Sarvam AI error: {str(e)}")

# --- MCP Server Setup ---
mcp = FastMCP(
    "VaaniDB MCP Server",
    description="Natural language to SQL database query system with Sarvam AI integration",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: query_database ---
QueryDatabaseDescription = RichToolDescription(
    description="Query the database using natural language",
    use_when="Use this when you need to get data from the database using plain English questions",
    side_effects="Executes SQL queries on the database and returns results",
)

@mcp.tool(description=QueryDatabaseDescription.model_dump_json())
async def query_database(
    question: Annotated[str, Field(description="Natural language question about the data")],
    max_tokens: Annotated[int, Field(description="Maximum tokens for AI response", default=100)] = 100
) -> str:
    """Query the database using natural language"""
    if not sarvam_client:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message="Sarvam AI not configured - check SARVAMAI_KEY"
        ))
    
    try:
        # Convert natural language to SQL
        sql_query = nl_to_sql_sarvam(question, max_tokens)
        
        # Execute the query
        conn = sqlite3.connect(DATABASE_PATH)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        results = result_df.to_dict(orient="records")
        
        return json.dumps({
            "success": True,
            "sql": sql_query,
            "result": results,
            "message": f"Query executed successfully. Found {len(results)} results."
        }, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Database query failed: {str(e)}"
        ))

# --- Tool: upload_csv_from_url ---
UploadCSVDescription = RichToolDescription(
    description="Upload CSV data from a URL to the database",
    use_when="Use this when you need to load data from a CSV URL into the database",
    side_effects="Replaces the existing data in the database with the new CSV data",
)

@mcp.tool(description=UploadCSVDescription.model_dump_json())
async def upload_csv_from_url(
    url: Annotated[AnyUrl, Field(description="URL of the CSV file")]
) -> str:
    """Upload CSV data from a URL to the database"""
    try:
        df = pd.read_csv(str(url))
        
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        
        return json.dumps({
            "success": True,
            "message": "CSV uploaded and stored in database successfully",
            "rows_uploaded": len(df),
            "columns": list(df.columns)
        }, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Failed to upload CSV: {str(e)}"
        ))

# --- Tool: get_database_info ---
DatabaseInfoDescription = RichToolDescription(
    description="Get basic information about the database",
    use_when="Use this to understand what tables and columns are available in the database",
    side_effects="None",
)

@mcp.tool(description=DatabaseInfoDescription.model_dump_json())
async def get_database_info() -> str:
    """Get basic information about the database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            table_info[table] = {
                "columns": columns,
                "row_count": row_count
            }
        
        conn.close()
        
        return json.dumps({
            "success": True,
            "tables": table_info,
            "message": "Database information retrieved successfully"
        }, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Failed to retrieve database information: {str(e)}"
        ))

# --- Tool: get_sample_data ---
SampleDataDescription = RichToolDescription(
    description="Get sample data from a table",
    use_when="Use this to see example data from a specific table",
    side_effects="None",
)

@mcp.tool(description=SampleDataDescription.model_dump_json())
async def get_sample_data(
    table_name: Annotated[str, Field(description="Name of the table", default="data")] = "data",
    limit: Annotated[int, Field(description="Number of rows to return", default=5)] = 5
) -> str:
    """Get sample data from a table"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        sample_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        conn.close()
        
        return json.dumps({
            "success": True,
            "sample_data": sample_df.to_dict(orient="records"),
            "columns": list(sample_df.columns),
            "message": f"Retrieved {len(sample_df)} sample rows from {table_name}"
        }, indent=2)
        
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"Failed to retrieve sample data: {str(e)}"
        ))

# --- Run MCP Server ---
async def main():
    print(f"🚀 Starting VaaniDB MCP Server on http://0.0.0.0:8086")
    print(f"Database path: {DATABASE_PATH}")
    print(f"Sarvam AI configured: {'✓' if sarvam_client else '✗'}")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())