import os
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sarvamai import SarvamAI  # Sarvam AI SDK

# Load environment variables from .env
load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./demo.db")

# Initialize Sarvam AI client
client = SarvamAI(api_subscription_key=SARVAMAI_KEY)

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# ---------------------- Utility ----------------------
def check_auth(req):
    """Check if Authorization header matches."""
    return req.headers.get("Authorization") == f"Bearer {AUTH_TOKEN}"

def nl_to_sql_sarvam(nl_question: str, max_tokens: int = 100) -> str:
    """Convert a natural language question into an SQL query."""
    prompt = (
        f"You are an assistant that converts natural language questions into SQL queries "
        f"for a SQLite table named 'data'.\n"
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

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Sarvam AI SQL API is running!"}), 200

@app.route("/upload", methods=["POST"])
def upload_csv():
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({"message": "CSV uploaded and stored in DB"}), 200

@app.route("/query", methods=["POST"])
def query_nl():
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

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

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql_query}), 400

    return jsonify({
        "sql": sql_query,
        "result": result_df.to_dict(orient="records")
    }), 200

# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)