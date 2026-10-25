import importlib.resources
import json
import sqlite3
import sys
from contextlib import closing

from ollama import chat, embed

DB_PATH = "store.db"
MODEL = "gemma4:e2b-mlx"  # warning: could generate incomplete SQL queries due being a smal model
EMBED_MODEL = "embeddinggemma-2:270m"

# =====================================================================
# 1. SAMPLE DATA SETUP (SQLite + sqlite-vector)
# =====================================================================


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    extension_name = {
        "darwin": "vector.dylib",
        "win32": "vector.dll",
    }.get(sys.platform, "vector.so")
    extension_path = importlib.resources.files("sqlite_vector.binaries") / extension_name
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(extension_path))
        finally:
            conn.enable_load_extension(False)
    except Exception:
        conn.close()
        raise
    return conn


def initialize_product_vectors(conn, dimension: int) -> None:
    conn.execute(
        "SELECT vector_init('product_docs', 'embedding', ?)",
        (f"dimension={dimension},type=FLOAT32,distance=COSINE",),
    )


def init_tables():
    with closing(get_connection()) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                category TEXT,
                units_sold INTEGER,
                unit_price REAL,
                sale_date DATE
            )
        """)
        sales_columns = {row[1] for row in cur.execute("PRAGMA table_info(sales)")}
        if "category" not in sales_columns:
            cur.execute("ALTER TABLE sales ADD COLUMN category TEXT")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()


def search_product_docs(query: str, top_k: int = 2) -> str:
    """Search warranty and product descriptions using vector similarity."""
    # Prefix query for EmbeddingGemma
    formatted_query = f"task: search query | query: {query}"
    res = embed(model=EMBED_MODEL, input=formatted_query)
    query_vector = json.dumps(res.embeddings[0])

    with closing(get_connection()) as conn:
        initialize_product_vectors(conn, len(res.embeddings[0]))
        cur = conn.cursor()
        cur.execute("""
            SELECT d.content, v.distance
            FROM vector_full_scan('product_docs', 'embedding', ?, ?) v
            JOIN product_docs d ON d.rowid = v.rowid
        """, (query_vector, top_k))
        rows = cur.fetchall()

    return json.dumps([{"content": r[0], "distance": r[1]} for r in rows])


def init_mock_data():
    # 1.1 SQLite Sales Database
    init_tables()
    docs = [
        "Aura Pro Wireless Headphones: Features active noise cancellation and 40-hour battery life. Warranty: 2 years limited warranty covering battery degradation and hardware defects. Accidental water damage is not covered.",
        "Titan Mechanical Keyboard: Features hot-swappable tactile switches, RGB backlight, and aluminum chassis. Warranty: 3 years warranty against switch failure and key chatter.",
        "Nova 4K Gaming Monitor: 144Hz IPS display with 1ms response time and HDR600. Warranty: 3-year zero-bright-dot guarantee. Free panel replacement if dead pixels occur within the first year."
    ]

    res = embed(model=EMBED_MODEL, input=docs)
    dimension = len(res.embeddings[0])

    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sales")
        cursor.executemany("""
            INSERT INTO sales (product_name, category, units_sold, unit_price, sale_date)
            VALUES (?, ?, ?, ?, ?)
        """, [
            ("Aura Pro Wireless Headphones", "Audio", 120, 199.99, "2026-03-01"),
            ("Aura Pro Wireless Headphones", "Audio", 85, 199.99, "2026-03-15"),
            ("Titan Mechanical Keyboard", "Peripherals", 40, 149.50, "2026-03-05"),
            ("Nova 4K Gaming Monitor", "Monitors", 15, 499.00, "2026-03-10"),
        ])

        initialize_product_vectors(conn, dimension)
        cursor.execute("DELETE FROM product_docs")
        cursor.executemany("""
            INSERT INTO product_docs (content, embedding)
            VALUES (?, vector_as_f32(?))
        """, [
            (doc, json.dumps(embedding))
            for doc, embedding in zip(docs, res.embeddings, strict=True)
        ])
        conn.commit()


# =====================================================================
# 2. TOOL DEFINITIONS (Standard Python Functions)
# =====================================================================

def query_sales_db(sql_query: str, parameters: list[str] | None = None) -> str:
    """
    Execute a read-only SQL query against the SQLite sales database.
    The database contains a 'sales' table with schema:
    sales(id INTEGER, product_name TEXT, category TEXT, units_sold INTEGER, unit_price REAL, sale_date DATE).

    Args:
        sql_query: A valid SQLite SELECT query.
        parameters: Values bound to positional placeholders in the query.
    """
    clean_sql = sql_query.strip()
    if not clean_sql.upper().startswith("SELECT"):
        return json.dumps({"error": "Only read-only SELECT queries are allowed."})

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(clean_sql, parameters or [])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return json.dumps([dict(zip(columns, row)) for row in rows])
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if "conn" in locals():
            conn.close()


# Tool dispatch registry
AVAILABLE_TOOLS = {
    "query_sales_db": query_sales_db,
    "search_product_docs": search_product_docs,
}

# =====================================================================
# 3. THE AGENT LOOP
# =====================================================================

SYSTEM_PROMPT = """You are an intelligent enterprise retail assistant.
You have access to two tools:
1. `query_sales_db` - Use this whenever you need to compute or retrieve sales statistics, revenue, or unit counts. It accepts `sql_query` and an optional `parameters` list for positional `?` placeholders. Always bind values through `parameters`; never interpolate or quote user-provided values into SQL.
2. `search_product_docs` - Use this when questions relate to product features, specifications, or warranty policies.

When filtering sales by a product name supplied by the user, use a case-insensitive prefix match with `product_name LIKE ? COLLATE NOCASE`. Put the user's product name followed by `%` in the corresponding `parameters` entry (for example, `sql_query: "SELECT SUM(units_sold) AS total_units, SUM(units_sold * unit_price) AS total_revenue FROM sales WHERE product_name LIKE ? COLLATE NOCASE"` and `parameters: ["Aura Pro%"]` for "Aura Pro"). This matches names that start with the supplied text, such as "Aura Pro Wireless Headphones".

If a query requires both sales data and product knowledge, call the tools sequentially before synthesizing your final answer.
Always respond clearly and accurately using the context retrieved from the tools.
"""

init_mock_data()

def run_agent(user_prompt: str):
    print(f"\nUser: {user_prompt}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    while True:
        # Step A: Model evaluates messages and decides whether to output text or a tool call
        response = chat(
            model=MODEL,
            messages=messages,
            tools=[query_sales_db, search_product_docs]  # Ollama parses function signatures automatically
        )

        # Append assistant's turn (message object) to memory
        messages.append(response.message)

        # Step B: Check if the model requested any tool calls
        if not response.message.tool_calls:
            # No tools called: the model produced the final answer
            print(f"\nAgent: {response.message.content}")
            break

        # Step C: Execute tool calls and append the results back to message history
        for tool_call in response.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            print(f"  [Tool Call] -> {fn_name}({fn_args})")

            if fn_name in AVAILABLE_TOOLS:
                tool_output = AVAILABLE_TOOLS[fn_name](**fn_args)
                print(f"  [Tool Response] -> {tool_output=}")
            else:
                tool_output = json.dumps({"error": f"Tool '{fn_name}' does not exist."})

            # Pass result back to the model as role="tool"
            messages.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": str(tool_output)
            })

# =====================================================================
# 4. TEST QUERIES
# =====================================================================

if __name__ == "__main__":
    # Test 1: Pure SQL Tool
    run_agent("How many units of the Aura Pro wireless headphones have we sold in total, and what was the revenue?")

    # Test 2: Pure RAG Tool
    run_agent("Does the warranty for the Aura Pro wireless headphones cover accidental water damage?")

    # Test 3: Compound Query (Calls both SQL and RAG)
    run_agent("Which product generated the most revenue, and what is its warranty period?")
