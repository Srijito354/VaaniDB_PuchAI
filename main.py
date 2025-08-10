#!/usr/bin/env python3
"""
VaaniDB MCP Server for Puch AI Hackathon
Converts your existing Flask SQL API into a Model Context Protocol server
Simple natural language to SQL conversion using Sarvam AI
"""

import logging
import sqlite3
import pandas as pd
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import io

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sarvamai import SarvamAI
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configuration from your existing app3.py
AUTH_TOKEN = os.getenv("PUCH_AUTH_TOKEN")
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")
PUCH_PHONE_NUMBER = os.getenv("PHONE_NUMBER")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize Sarvam AI client
if SARVAMAI_KEY:
    sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_KEY)
else:
    sarvam_client = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 100

class UploadURLRequest(BaseModel):
    url: str

def nl_to_sql_sarvam(nl_question: str, max_tokens: int = 100) -> str:
    """Convert natural language to SQL using Sarvam AI - same as your original function"""
    if not sarvam_client:
        raise ValueError("Sarvam AI client not initialized - check SARVAMAI_KEY")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get column names (only predefined SQL we need)
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

async def verify_auth_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting VaaniDB MCP Server...")
    logger.info(f"Database path: {DATABASE_PATH}")
    logger.info(f"Sarvam AI configured: {'✓' if sarvam_client else '✗'}")
    yield
    logger.info("Shutting down VaaniDB MCP Server...")

# Create FastAPI app
app = FastAPI(
    title="VaaniDB MCP Server",
    description="Natural language to SQL database query system with Sarvam AI integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Protocol handler
class MCPHandler:
    """Simple MCP protocol handler"""
    
    def __init__(self):
        self.tools = {}
    
    def tool(self, func=None):
        """Decorator to register MCP tools"""
        def decorator(f):
            self.tools[f.__name__] = f
            return f
        
        if func is None:
            # Called as @mcp.tool()
            return decorator
        else:
            # Called as @mcp.tool
            return decorator(func)
    
    def list_tools(self):
        """List all available tools"""
        tools_list = []
        for name, func in self.tools.items():
            # Extract docstring and parameters
            doc = func.__doc__ or "No description available"
            tools_list.append({
                "name": name,
                "description": doc.strip().split('\n')[0] if doc else "No description"
            })
        return tools_list
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool with arguments"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            result = self.tools[tool_name](**arguments)
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

# Initialize MCP handler
mcp = MCPHandler()

@mcp.tool
async def validate() -> str:
    return PUCH_PHONE_NUMBER

@mcp.tool()
def query_database(question: str, max_tokens: int = 100) -> Dict[str, Any]:
    """
    Query the database using natural language
    
    Args:
        question: Natural language question about the data
        max_tokens: Maximum tokens for AI response (default: 100)
    
    Returns:
        Dictionary with SQL query and results
    """
    if not sarvam_client:
        return {
            "success": False,
            "error": "Sarvam AI not configured - check SARVAMAI_KEY",
            "sql": "",
            "result": []
        }
    
    try:
        # Convert natural language to SQL
        sql_query = nl_to_sql_sarvam(question, max_tokens)
        
        # Execute the query
        conn = sqlite3.connect(DATABASE_PATH)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        results = result_df.to_dict(orient="records")
        
        return {
            "success": True,
            "sql": sql_query,
            "result": results,
            "message": f"Query executed successfully. Found {len(results)} results."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sql": sql_query if 'sql_query' in locals() else "",
            "result": []
        }

@mcp.tool()
def upload_csv_from_url(url: str) -> Dict[str, Any]:
    """
    Upload CSV data from a URL to the database
    
    Args:
        url: URL of the CSV file
    
    Returns:
        Dictionary with upload status
    """
    try:
        df = pd.read_csv(url)
        
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        
        return {
            "success": True,
            "message": "CSV uploaded and stored in database successfully",
            "rows_uploaded": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to upload CSV from URL: {url}"
        }

@mcp.tool()
def get_database_info() -> Dict[str, Any]:
    """
    Get basic information about the database
    
    Returns:
        Dictionary with database information
    """
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
        
        return {
            "success": True,
            "tables": table_info,
            "message": "Database information retrieved successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tables": {},
            "message": "Failed to retrieve database information"
        }

@mcp.tool()
def get_sample_data(table_name: str = "data", limit: int = 5) -> Dict[str, Any]:
    """
    Get sample data from a table
    
    Args:
        table_name: Name of the table (default: "data")
        limit: Number of rows to return (default: 5)
    
    Returns:
        Dictionary with sample data
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        sample_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        conn.close()
        
        return {
            "success": True,
            "sample_data": sample_df.to_dict(orient="records"),
            "columns": list(sample_df.columns),
            "message": f"Retrieved {len(sample_df)} sample rows from {table_name}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sample_data": [],
            "message": f"Failed to retrieve sample data from {table_name}"
        }

# MCP Protocol endpoints
@app.post("/mcp")
async def mcp_endpoint(request: dict, token: str = Depends(verify_auth_token)):
    """MCP protocol endpoint"""
    try:
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {
                "tools": mcp.list_tools()
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            result = mcp.call_tool(tool_name, arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
            }
        
        else:
            return {"error": f"Unknown method: {method}"}
    
    except Exception as e:
        return {"error": f"MCP request failed: {str(e)}"}

@app.get("/mcp")
async def mcp_info():
    """MCP server information"""
    return {
        "name": "VaaniDB",
        "version": "1.0.0",
        "description": "Natural language to SQL database query system",
        "tools": mcp.list_tools()
    }

# REST API endpoints (maintaining compatibility with your existing app3.py)
@app.get("/")
async def home():
    """Root endpoint"""
    return {
        "message": "VaaniDB MCP Server is running!",
        "version": "1.0.0",
        "features": ["Natural Language to SQL", "CSV Upload", "MCP Protocol"],
        "mcp_endpoint": "/mcp"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_path": DATABASE_PATH,
        "sarvam_ai_configured": sarvam_client is not None
    }

@app.get("/validate")
async def validate():
    """Required by Puch AI to verify server ownership"""
    if not PUCH_PHONE_NUMBER:
        return {"error": "Phone number not configured"}, 500
    
    return {
        "phone_number": PUCH_PHONE_NUMBER,
        "status": "validated"
    }

@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    token: str = Depends(verify_auth_token)
):
    """Upload CSV file endpoint (same as your original app3.py)"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()

        return {"message": "CSV uploaded and stored in DB"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-url")
async def upload_csv_from_url_endpoint(
    request: UploadURLRequest,
    token: str = Depends(verify_auth_token)
):
    """Upload CSV from URL endpoint (same as your original app3.py)"""
    try:
        df = pd.read_csv(request.url)
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        
        return {"message": "CSV uploaded from URL and stored in DB"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load CSV: {str(e)}")

@app.post("/query")
async def query_nl(
    request: QueryRequest,
    token: str = Depends(verify_auth_token)
):
    """Natural language query endpoint (same as your original app3.py)"""
    if not sarvam_client:
        raise HTTPException(status_code=500, detail="No SARVAMAI_KEY set")

    try:
        sql_query = nl_to_sql_sarvam(request.question, request.max_tokens)
    except ValueError as err:
        raise HTTPException(status_code=500, detail=str(err))

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        result = result_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "sql": sql_query})
    finally:
        conn.close()

    return {"sql": sql_query, "result": result}

if __name__ == "__main__":
    logger.info(f"Starting VaaniDB MCP Server on {HOST}:{PORT}")
    logger.info(f"Auth token configured: {'✓' if AUTH_TOKEN else '✗'}")
    logger.info(f"Sarvam AI configured: {'✓' if sarvam_client else '✗'}")
    logger.info(f"Database path: {DATABASE_PATH}")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )