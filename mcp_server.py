import os
import sqlite3
import pandas as pd
import json
import traceback
from typing import Annotated
from pydantic import Field
from fastmcp import FastMCP

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment loaded")
except Exception as e:
    print(f"⚠️ dotenv error: {e}")

# Environment variables
PUCH_AUTH_TOKEN = os.getenv("PUCH_AUTH_TOKEN")
PHONE_NUMBER = os.getenv("PHONE_NUMBER") 
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")

print(f"📱 Phone: {'✅' if PHONE_NUMBER else '❌'}")
print(f"🔐 Puch Auth: {'✅' if PUCH_AUTH_TOKEN else '❌'}")
print(f"🤖 Sarvam Key: {'✅' if SARVAMAI_KEY else '❌'}")
print(f"📁 Database: {DATABASE_PATH}")

# Initialize FastMCP server
mcp = FastMCP("nl_sql_assistant")

# Try to initialize Sarvam AI if key is available
client = None
if SARVAMAI_KEY:
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=SARVAMAI_KEY)
        print("✅ Sarvam AI initialized")
    except Exception as e:
        print(f"⚠️ Sarvam AI error: {e}")

def simple_nl_to_sql(question: str, columns: list) -> str:
    """Fallback SQL generation when AI is not available"""
    question_lower = question.lower()
    
    if "count" in question_lower or "how many" in question_lower:
        return "SELECT COUNT(*) as count FROM data"
    elif "average" in question_lower or "avg" in question_lower:
        # Try to find numeric columns
        numeric_cols = [col for col in columns if col.lower() not in ['name', 'product', 'category', 'description', 'id']]
        if numeric_cols:
            return f"SELECT AVG({numeric_cols[0]}) as average FROM data"
        return "SELECT COUNT(*) as count FROM data"
    elif "maximum" in question_lower or "max" in question_lower:
        numeric_cols = [col for col in columns if col.lower() not in ['name', 'product', 'category', 'description', 'id']]
        if numeric_cols:
            return f"SELECT MAX({numeric_cols[0]}) as maximum FROM data"
        return "SELECT * FROM data LIMIT 1"
    elif "sum" in question_lower or "total" in question_lower:
        numeric_cols = [col for col in columns if col.lower() not in ['name', 'product', 'category', 'description', 'id']]
        if numeric_cols:
            return f"SELECT SUM({numeric_cols[0]}) as total FROM data"
        return "SELECT COUNT(*) as count FROM data"
    else:
        return "SELECT * FROM data LIMIT 10"

@mcp.tool(description="Test the server and get debug information")
async def debug_info() -> str:
    """Get debug information about the server status"""
    info = {
        "server_status": "✅ Running",
        "timestamp": pd.Timestamp.now().isoformat(),
        "configuration": {
            "phone_configured": "✅" if PHONE_NUMBER else "❌",
            "puch_auth_configured": "✅" if PUCH_AUTH_TOKEN else "❌", 
            "sarvam_configured": "✅" if SARVAMAI_KEY else "❌",
            "database_path": DATABASE_PATH
        }
    }
    
    # Test database connection
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        info["database"] = {
            "status": "✅ Connected",
            "tables": [table[0] for table in tables],
            "has_data_table": "data" in [table[0] for table in tables]
        }
    except Exception as e:
        info["database"] = {
            "status": f"❌ Error: {str(e)}"
        }
    
    # Test Sarvam AI if configured
    if client:
        try:
            resp = client.chat.completions(
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=5,
                temperature=0
            )
            info["sarvam_ai"] = {"status": "✅ Connected"}
        except Exception as e:
            info["sarvam_ai"] = {"status": f"❌ Error: {str(e)}"}
    else:
        info["sarvam_ai"] = {"status": "❌ Not configured"}
    
    return json.dumps(info, indent=2)

@mcp.tool(description="Upload CSV from public URL")
async def upload_csv(
    url: Annotated[str, Field(description="Public URL to CSV file")]
) -> str:
    """Upload CSV from URL and store in database"""
    try:
        print(f"📥 Loading CSV from: {url}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(url, encoding=encoding)
                print(f"✅ Loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            return json.dumps({
                "status": "error",
                "message": "Could not read CSV with any encoding"
            })
        
        print(f"📊 CSV Info: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Store in database
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        
        return json.dumps({
            "status": "success",
            "message": "CSV uploaded successfully! 🎉",
            "data_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "sample_data": df.head(2).to_dict(orient="records")
            }
        }, indent=2)
        
    except Exception as e:
        error_msg = {
            "status": "error", 
            "message": f"Failed to upload CSV: {str(e)}",
            "url": url
        }
        print(f"❌ Upload failed: {error_msg}")
        return json.dumps(error_msg)

@mcp.tool(description="Ask natural language questions about your data")
async def query_data(
    question: Annotated[str, Field(description="Natural language question about the data")]
) -> str:
    """Convert natural language to SQL and return results"""
    try:
        print(f"❓ Question: {question}")
        
        # Check if data table exists
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
        if not cursor.fetchone():
            conn.close()
            return json.dumps({
                "status": "error",
                "message": "❌ No data found. Please upload a CSV first using upload_csv!"
            })
        
        # Get table schema
        cursor.execute("PRAGMA table_info(data)")
        schema_info = cursor.fetchall()
        columns = [col[1] for col in schema_info]
        print(f"📋 Available columns: {columns}")
        
        # Generate SQL query
        if client and SARVAMAI_KEY:
            try:
                # Get sample data for better context
                cursor.execute("SELECT * FROM data LIMIT 2")
                sample_rows = cursor.fetchall()
                
                prompt = f"""You are a SQL expert. Convert this natural language question to a SQLite query.

Database Information:
- Table name: data
- Columns: {', '.join(columns)}
- Sample data: {sample_rows}

Question: {question}

Important:
- Return ONLY the SQL query
- Use SQLite syntax
- No explanations, no markdown
- Query should be ready to execute

SQL Query:"""

                resp = client.chat.completions(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                
                sql_query = resp.choices[0].message.content.strip()
                # Clean up response
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                # Remove any explanatory text
                lines = sql_query.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('--'):
                        clean_lines.append(line)
                sql_query = ' '.join(clean_lines)
                
                print(f"🤖 AI generated: {sql_query}")
                
            except Exception as ai_error:
                print(f"⚠️ AI failed: {ai_error}, using fallback")
                sql_query = simple_nl_to_sql(question, columns)
                print(f"🔄 Fallback SQL: {sql_query}")
        else:
            sql_query = simple_nl_to_sql(question, columns)
            print(f"🔄 Using fallback SQL: {sql_query}")
        
        # Execute the query
        result_df = pd.read_sql_query(sql_query, conn)
        result = result_df.to_dict(orient="records")
        conn.close()
        
        response = {
            "status": "success",
            "question": question,
            "sql_query": sql_query,
            "results": result,
            "total_results": len(result),
            "ai_powered": bool(client and SARVAMAI_KEY)
        }
        
        print(f"✅ Query executed successfully, returned {len(result)} results")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
            
        error_response = {
            "status": "error",
            "question": question,
            "error_message": str(e),
            "sql_attempted": sql_query if 'sql_query' in locals() else "None"
        }
        print(f"❌ Query failed: {error_response}")
        return json.dumps(error_response, indent=2)

@mcp.tool(description="Get information about the current dataset")
async def get_data_info() -> str:
    """Show information about the currently loaded dataset"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if data table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
        if not cursor.fetchone():
            conn.close()
            return json.dumps({
                "status": "info",
                "message": "❌ No dataset loaded. Use upload_csv to load data first!"
            })
        
        # Get table schema
        cursor.execute("PRAGMA table_info(data)")
        schema_info = cursor.fetchall()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM data")
        row_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute("SELECT * FROM data LIMIT 5")
        sample_rows = cursor.fetchall()
        column_names = [col[1] for col in schema_info]
        
        conn.close()
        
        return json.dumps({
            "status": "success",
            "dataset_info": {
                "table_name": "data",
                "total_rows": row_count,
                "total_columns": len(column_names),
                "columns": [{"name": col[1], "type": col[2]} for col in schema_info],
                "sample_data": [dict(zip(column_names, row)) for row in sample_rows]
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error getting data info: {str(e)}"
        })

@mcp.tool(description="Load demo data for testing")
async def load_demo() -> str:
    """Load sample sales data for demonstration"""
    try:
        # Create comprehensive sample data
        sample_data = {
            'product_id': [1, 2, 3, 4, 5, 6],
            'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam'],
            'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Audio', 'Electronics'],
            'price': [75000, 1500, 3500, 25000, 8000, 4500],
            'quantity_sold': [15, 120, 85, 20, 60, 45],
            'rating': [4.5, 4.2, 4.7, 4.3, 4.6, 4.1],
            'in_stock': [True, True, False, True, True, True]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Store in database
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        
        return json.dumps({
            "status": "success",
            "message": "🎉 Demo data loaded successfully!",
            "data_loaded": sample_data,
            "suggestions": [
                "How many products are there?",
                "What is the average price?",
                "Which product has the highest rating?",
                "Show me all Electronics products",
                "What's the total revenue if all products sell?"
            ]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to load demo data: {str(e)}"
        })

@mcp.tool(description="Validate phone number for Puch AI")
async def validate() -> str:
    """Validate phone number for Puch AI integration"""
    if not PHONE_NUMBER:
        return json.dumps({
            "status": "error", 
            "message": "❌ Phone number not configured in environment"
        })
    
    return json.dumps({
        "phone_number": PHONE_NUMBER,
        "status": "validated",
        "message": "✅ Phone number validated for Puch AI"
    })

if __name__ == "__main__":
    print("🚀 Starting Natural Language SQL Assistant...")
    print("📋 Available tools:")
    print("  • debug_info - Check server status")
    print("  • upload_csv - Upload CSV from URL") 
    print("  • query_data - Ask questions in natural language")
    print("  • get_data_info - Show dataset information")
    print("  • load_demo - Load sample data")
    print("  • validate - Puch AI validation")
    
    # Run the FastMCP server (it handles uvicorn internally)
    #mcp.run(port=8000, host="0.0.0.0")
    #mcp.run()
    port = int(os.environ.get("PORT", 5000))
    mcp.run(port=port, debug=False)