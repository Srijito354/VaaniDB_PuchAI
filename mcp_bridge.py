import os
import sqlite3
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

# Load environment variables
AUTH_TOKEN = os.getenv("PUCH_AUTH_TOKEN")
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")

# Initialize Sarvam AI client
client = SarvamAI(api_subscription_key=SARVAMAI_KEY)

app = Flask(__name__)
CORS(app)

def check_auth(req):
    auth_header = req.headers.get("Authorization", "")
    return auth_header == f"Bearer {AUTH_TOKEN}"

def nl_to_sql_sarvam(nl_question: str) -> str:
    prompt = f"""You are an assistant that converts natural language questions into SQL queries for a SQLite table named 'data'.

Question: {nl_question}

Only return the SQL query, nothing else."""
    
    try:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Sarvam AI error: {str(e)}")

# MCP Protocol Handler
@app.route("/", methods=["POST"])
def mcp_handler():
    """Handle MCP JSON-RPC requests"""
    if not check_auth(request):
        return jsonify({
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "Unauthorized"},
            "id": request.json.get("id") if request.json else None
        }), 401

    try:
        data = request.json
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")

        if method == "tools/list":
            return jsonify({
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {
                            "name": "upload_csv",
                            "description": "Upload CSV from public URL",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Public URL to CSV file"}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "query_data", 
                            "description": "Ask natural language questions about your data",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string", "description": "Natural language question"}
                                },
                                "required": ["question"]
                            }
                        }
                    ]
                },
                "id": request_id
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "upload_csv":
                url = arguments.get("url")
                try:
                    df = pd.read_csv(url)
                    conn = sqlite3.connect(DATABASE_PATH)
                    df.to_sql("data", conn, if_exists="replace", index=False)
                    conn.close()
                    
                    result = f"✅ CSV uploaded successfully!\n📊 {len(df)} rows, {len(df.columns)} columns\n📋 Columns: {list(df.columns)}"
                    
                except Exception as e:
                    result = f"❌ Error uploading CSV: {str(e)}"

                return jsonify({
                    "jsonrpc": "2.0",
                    "result": {"content": [{"type": "text", "text": result}]},
                    "id": request_id
                })

            elif tool_name == "query_data":
                question = arguments.get("question")
                try:
                    # Check if data exists
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data'")
                    if not cursor.fetchone():
                        conn.close()
                        return jsonify({
                            "jsonrpc": "2.0",
                            "result": {"content": [{"type": "text", "text": "❌ No data found. Please upload a CSV first!"}]},
                            "id": request_id
                        })

                    # Generate SQL with Sarvam AI
                    sql_query = nl_to_sql_sarvam(question)
                    
                    # Execute query
                    result_df = pd.read_sql_query(sql_query, conn)
                    conn.close()
                    
                    # Format result
                    result_text = f"🤖 SQL Generated: {sql_query}\n\n📊 Results ({len(result_df)} rows):\n"
                    if len(result_df) > 0:
                        result_text += result_df.to_string(index=False)
                    else:
                        result_text += "No results found."
                        
                except Exception as e:
                    result_text = f"❌ Error: {str(e)}"

                return jsonify({
                    "jsonrpc": "2.0", 
                    "result": {"content": [{"type": "text", "text": result_text}]},
                    "id": request_id
                })

            else:
                return jsonify({
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    "id": request_id
                })

        else:
            return jsonify({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
                "id": request_id
            })

    except Exception as e:
        return jsonify({
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            "id": request.json.get("id") if request.json else None
        }), 500

# Health check for testing
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "MCP Bridge Running", "message": "Ready for PuchAI integration"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)