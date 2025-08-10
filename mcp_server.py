import os
import sqlite3
import pandas as pd
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from pydantic import Field, BaseModel
from sarvamai import SarvamAI

# --- Load Environment Variables ---
load_dotenv()

AUTH_TOKEN = os.getenv("PUCH_AUTH_TOKEN")
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")
PUCH_PHONE_NUMBER = os.getenv("PHONE_NUMBER")  # Format: 919876543210

assert AUTH_TOKEN, "Missing PUCH_AUTH_TOKEN in .env"
assert PUCH_PHONE_NUMBER, "Missing PHONE_NUMBER in .env"
assert SARVAMAI_KEY, "Missing SARVAMAI_KEY in .env"

# --- Initialize Services ---
client = SarvamAI(api_subscription_key=SARVAMAI_KEY)

# --- Auth Provider (Puch AI Requirement) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        return AccessToken(
            token=token,
            client_id="puch-client",
            scopes=["*"],
            expires_at=None,
        ) if token == self.token else None

# --- Tool Descriptions ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Initialize MCP Server ---
mcp = FastMCP(
    "NL2SQL Assistant",
    auth=SimpleBearerAuthProvider(AUTH_TOKEN),
)

# --- Core NL2SQL Function ---
async def generate_sql(question: str, columns: list) -> str:
    """Convert natural language to SQL using Sarvam AI"""
    prompt = f"""
    You are a SQL expert. Convert this question to SQLite SQL.
    Database: Table 'data' with columns: {', '.join(columns)}
    
    Rules:
    1. Return ONLY the SQL query
    2. Use SQLite syntax
    3. Never use columns not in: {columns}
    4. Handle NULL values properly
    
    Question: {question}
    SQL Query:
    """
    
    try:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        sql = resp.choices[0].message.content.strip()
        return sql.replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        raise McpError(ErrorData(
            code="AI_ERROR",
            message=f"AI service failed: {str(e)}"
        ))

# --- Required Validation Tool ---
@mcp.tool
async def validate() -> str:
    """Puch AI ownership verification"""
    return PUCH_PHONE_NUMBER

# --- SQL Query Tool ---
SQLToolDesc = RichToolDescription(
    description="Convert questions to SQL and return results",
    use_when="Asking about data in the database",
    side_effects="Executes SQL on the connected database"
)

@mcp.tool(description=SQLToolDesc.model_dump_json())
async def query(
    question: Annotated[str, Field(description="Natural language question about the data")]
) -> dict:
    """Main NL2SQL endpoint"""
    try:
        # Get schema
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
        if not cursor.fetchone():
            raise McpError(ErrorData(
                code="NO_DATA",
                message="No data found. Upload a CSV first using /upload_csv"
            ))
        
        # Get columns
        cursor.execute("PRAGMA table_info(data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Generate and execute SQL
        sql_query = await generate_sql(question, columns)
        result = pd.read_sql_query(sql_query, conn).to_dict(orient="records")
        
        return {
            "sql": sql_query,
            "results": result,
            "phone_number": PUCH_PHONE_NUMBER,
            "status": "success"
        }
        
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(
            code="QUERY_FAILED",
            message=f"Database error: {str(e)}"
        ))
    finally:
        conn.close()

# --- Data Upload Tool ---
UploadToolDesc = RichToolDescription(
    description="Upload CSV data to query",
    use_when="Need to load new dataset",
    side_effects="Replaces existing data"
)

@mcp.tool(description=UploadToolDesc.model_dump_json())
async def upload_csv(
    file_url: Annotated[str, Field(description="Public URL of CSV file")]
) -> dict:
    """Data ingestion endpoint"""
    try:
        # Supported encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_url, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise McpError(ErrorData(
                code="ENCODING_ERROR",
                message="Could not read CSV with any standard encoding"
            ))
        
        # Store data
        with sqlite3.connect(DATABASE_PATH) as conn:
            df.to_sql("data", conn, if_exists="replace", index=False)
        
        return {
            "message": "Data uploaded successfully",
            "columns": list(df.columns),
            "sample_data": df.head(2).to_dict(orient="records"),
            "phone_number": PUCH_PHONE_NUMBER
        }
        
    except Exception as e:
        raise McpError(ErrorData(
            code="UPLOAD_FAILED",
            message=f"Upload failed: {str(e)}"
        ))

# --- Server Startup ---
async def main():
    print(f"🚀 NL2SQL MCP Server | Phone: {PUCH_PHONE_NUMBER}")
    print("Available tools:")
    print("- /validate (Puch AI verification)")
    print("- /query (Ask questions about your data)")
    print("- /upload_csv (Load new datasets)")
    
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())