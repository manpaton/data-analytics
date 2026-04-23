import sqlite3

def get_connection():
    return sqlite3.connect("users.db")


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            gender TEXT,
            first_name TEXT,
            last_name TEXT,
            nationality TEXT,
            age INTEGER,
            age_group TEXT,
            email_domain TEXT,
            dob_date TEXT,
            loaded_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS etl_control (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()