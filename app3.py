import os
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for web compatibility
from dotenv import load_dotenv
from sarvamai import SarvamAI  # Sarvam AI SDK

load_dotenv()

# Load environment variables - Fixed to match your .env file
AUTH_TOKEN = os.getenv("PUCH_AUTH_TOKEN")  # Changed to match your .env
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")
PUCH_PHONE_NUMBER = os.getenv("PHONE_NUMBER")

# Initialize Sarvam AI client
client = SarvamAI(api_subscription_key=SARVAMAI_KEY)

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Utility: check auth
#def check_auth(req):
    #return req.headers.get("Authorization") == f"Bearer {AUTH_TOKEN}"

def nl_to_sql_sarvam(nl_question: str, max_tokens: int = 100) -> str:

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("PRAGMA table_info(data)")
    columns = [col[1] for col in cursor.fetchall()]  # Extract column names
    conn.close()

    prompt = (
        f"You are an assistant that converts natural language questions into SQL queries for a SQLite table named 'data'.\n"
        f"Available columns: {', '.join(columns)}"
        f"Question: {nl_question}\n"
        f"Only return the SQL query, nothing else."
    )
    try:
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        sql_query = resp.choices[0].message.content.strip()
        return sql_query
    except Exception as e:
        raise ValueError(f"Sarvam AI error: {str(e)}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sarvam AI SQL API is running!"})

@app.route("/upload", methods=["POST"])
def upload_csv():
    #if not check_auth(request):
        #return jsonify({"error": "Unauthorized"}), 401

    #if "file" not in request.files:
        #return jsonify({"error": "No file uploaded"}), 400 This shouldn't have been included, but it does for some reason. Sheesh!

    file = request.files["file"]
    df = pd.read_csv(file)

    conn = sqlite3.connect(DATABASE_PATH)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()

    return jsonify({"message": "CSV uploaded and stored in DB"}), 200

@app.route("/upload-url", methods=["POST"])  # Added for easier testing
def upload_csv_from_url():
    #if not check_auth(request):
    #return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        df = pd.read_csv(url)
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
        return jsonify({"message": "CSV uploaded from URL and stored in DB"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load CSV: {str(e)}"}), 400

@app.route("/query", methods=["POST"])
def query_nl():
    #if not check_auth(request):
    #return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not SARVAMAI_KEY:
        return jsonify({"error": "No SARVAMAI_KEY set"}), 500

    try:
        sql_query = nl_to_sql_sarvam(question)
    except ValueError as err:
        return jsonify({"error": str(err)}), 500

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        result = result_df.to_dict(orient="records")
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql_query}), 400
    finally:
        conn.close()

    return jsonify({"sql": sql_query, "result": result}), 200

@app.route("/validate", methods=["GET"])
def validate():
    """Required by Puch AI to verify server ownership"""
    phone_number = os.getenv("PUCH_PHONE_NUMBER")
    
    if not phone_number:
        return jsonify({"error": "Phone number not configured"}), 500
    
    return jsonify({
        "phone_number": phone_number,
        "status": "validated"
    }), 200

@app.route("/summarize", methods=["POST"])
def summarize_result():
    data = request.json
    question = data.get("question")
    result = data.get("result")

    if not question or not isinstance(result, list):
        return jsonify({"error": "Missing question or result"}), 400

    try:
        prompt = (
            f"You are an assistant. Given the question:\n"
            f"{question}\n"
            f"And the following query result rows:\n"
            f"{result}\n"
            f"Write a concise and clear natural language answer."
        )
        resp = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2
        )
        answer = resp.choices[0].message.content.strip()
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"Sarvam AI error: {str(e)}"}), 500

if __name__ == "__main__":
    # Only change: use Railway's PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)