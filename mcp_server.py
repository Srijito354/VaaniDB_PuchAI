import os
import sqlite3
import pandas as pd
from fastmcp import FastMCP
from pydantic import Field
from typing import Annotated
from sarvamai import SarvamAI
import json
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Environment variables
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")

# Initialize Sarvam AI client
client = SarvamAI(api_subscription_key=SARVAMAI_KEY)

# Initialize MCP server - try different approaches
try:
    # Option 1: With positional argument
    mcp = FastMCP("nl_sql_tool")
except TypeError:
    try:
        # Option 2: No arguments
        mcp = FastMCP()
    except TypeError:
        try:
            # Option 3: With name parameter
            mcp = FastMCP(name="nl_sql_tool")
        except TypeError:
            # Option 4: Check what parameters are actually accepted
            import inspect
            sig = inspect.signature(FastMCP.__init__)
            print(f"FastMCP.__init__ parameters: {list(sig.parameters.keys())}")
            # Default fallback
            mcp = FastMCP()

@mcp.tool(description="Upload CSV from public URL and store in SQLite DB")
async def upload_csv_url(
    file_url: Annotated[str, Field(description="Public URL to CSV file")]
) -> str:
    try:
        print(f"Attempting to load CSV from: {file_url}")
        df = pd.read_csv(file_url)
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        error_msg = f"Failed to load CSV from URL: {str(e)}"
        print(error_msg)
        return error_msg
    
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        df.to_sql("data", conn, if_exists="replace", index=False)
        success_msg = f"CSV uploaded successfully!\nShape: {df.shape}\nColumns: {list(df.columns)}"
        print(success_msg)
        return success_msg
    except Exception as e:
        error_msg = f"Database error: {str(e)}"
        print(error_msg)
        return error_msg
    finally:
        conn.close()

@mcp.tool(description="Ask a natural language question about the uploaded CSV data")
async def ask_data_question(
    question: Annotated[str, Field(description="Natural language question")]
) -> str:
    print(f"Processing question: {question}")
    
    # Check if data table exists and get schema
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
        if not cursor.fetchone():
            return "❌ No data table found. Please upload a CSV first using upload_csv_url."
        
        # Get column information
        cursor.execute("PRAGMA table_info(data)")
        schema_info = cursor.fetchall()
        columns = [col[1] for col in schema_info]
        schema_text = ", ".join(columns)
        print(f"Table columns: {schema_text}")
        
    except Exception as e:
        conn.close()
        error_msg = f"Database error: {str(e)}"
        print(error_msg)
        return error_msg
    
    # Generate SQL using Sarvam AI
    prompt = (
        f"You are a SQL expert. Convert this natural language question to a SQLite query.\n"
        f"Database table: 'data'\n"
        f"Available columns: {schema_text}\n"
        f"User question: {question}\n\n"
        f"Rules:\n"
        f"- Return ONLY the SQL query\n"
        f"- Use proper SQLite syntax\n"
        f"- No explanations, no markdown, no code blocks\n"
        f"- Query should be ready to execute\n"
    )
    
    try:
        print("Generating SQL with Sarvam AI...")
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        sql_query = resp.choices[0].message.content.strip()
        
        # Clean up any markdown formatting
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Remove any extra text that might be included
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines)
        print(f"Generated SQL: {sql_query}")
        
    except Exception as e:
        conn.close()
        error_msg = f"Sarvam AI error: {str(e)}"
        print(error_msg)
        return error_msg
    
    # Execute the SQL query
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        result = result_df.to_dict(orient="records")
        
        response = {
            "question": question,
            "sql_query": sql_query,
            "result": result,
            "row_count": len(result),
            "status": "success"
        }
        
        print(f"Query executed successfully. Returned {len(result)} rows.")
        return json.dumps(response, indent=2, default=str)
        
    except Exception as e:
        error_response = {
            "question": question,
            "sql_query": sql_query,
            "error": str(e),
            "status": "error"
        }
        print(f"SQL execution error: {str(e)}")
        return json.dumps(error_response, indent=2, default=str)
    finally:
        conn.close()

@mcp.tool(description="Show current table schema and basic info")
async def show_table_info() -> str:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
        if not cursor.fetchone():
            return "❌ No data table found. Upload a CSV first using upload_csv_url."
        
        # Get schema info
        cursor.execute("PRAGMA table_info(data)")
        schema_info = cursor.fetchall()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM data")
        row_count = cursor.fetchone()[0]
        
        # Get first few rows as sample
        cursor.execute("SELECT * FROM data LIMIT 3")
        sample_rows = cursor.fetchall()
        column_names = [col[1] for col in schema_info]
        
        info = {
            "table_name": "data",
            "columns": [{"name": col[1], "type": col[2], "not_null": bool(col[3])} for col in schema_info],
            "total_rows": row_count,
            "sample_data": [dict(zip(column_names, row)) for row in sample_rows]
        }
        
        print(f"Table info retrieved: {row_count} rows, {len(info['columns'])} columns")
        return json.dumps(info, indent=2, default=str)
        
    except Exception as e:
        error_msg = f"Error getting table info: {str(e)}"
        print(error_msg)
        return error_msg
    finally:
        conn.close()

@mcp.tool(description="Test the database connection and Sarvam AI")
async def test_connection() -> str:
    results = {}
    
    # Test database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        results["database"] = "✅ Connected"
    except Exception as e:
        results["database"] = f"❌ Error: {str(e)}"
    
    # Test Sarvam AI
    try:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        results["sarvam_ai"] = "✅ Connected"
    except Exception as e:
        results["sarvam_ai"] = f"❌ Error: {str(e)}"
    
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    print("🚀 Starting NL-SQL MCP Server...")
    print(f"📁 Database: {DATABASE_PATH}")
    print(f"🤖 SarvamAI configured: {'Yes' if SARVAMAI_KEY else 'No'}")
    print(f"🔧 FastMCP initialized: {type(mcp)}")
    
    try:
        uvicorn.run(
            mcp,  # Pass the mcp object directly
            host="0.0.0.0",
            port=5000,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start with mcp object, trying module string...")
        uvicorn.run(
            "mcp_server:mcp",
            host="0.0.0.0",
            port=5000,
            log_level="info",
            reload=True
        )