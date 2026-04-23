import requests as rq
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from db import get_connection, init_db
url = "https://randomuser.me/api/?results=20"

# ---------------- LOGGING ----------------
logger = logging.getLogger("etl_logger")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "etl_pipeline.log", maxBytes=1_000_000, backupCount=3
)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------- EXTRACT ----------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_data():
    response = rq.get(url, timeout=5)
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}")
    return response.json()


def extract(existing_emails=None):
    logger.info("Starting extract")

    data = fetch_data()
    users = data.get("results", [])
    df = pd.json_normalize(users)

    if existing_emails:
        before = len(df)
        df = df[~df["email"].isin(existing_emails)]
        logger.warning(f"Filtered duplicates: {before - len(df)}")

    logger.info(f"Extracted rows: {len(df)}")
    return df


# ---------------- TRANSFORM ----------------
def transform(df):
    logger.info("Starting transform")

    if df.empty:
        return df

    df = df.copy()

    df["first_name"] = df["name.first"]
    df["last_name"] = df["name.last"]

    df["age"] = df["dob.age"]
    df["dob_date"] = pd.to_datetime(df["dob.date"], errors="coerce")

    conditions = [
        df["age"] < 18,
        (df["age"] <= 30),
        (df["age"] <= 60),
        (df["age"] > 60)
    ]

    choices = ["Child", "Young Adult", "Adult", "Senior"]

    df["age_group"] = pd.Series(pd.NA, index=df.index)
    df.loc[df["age"] < 18, "age_group"] = "Child"
    df.loc[(df["age"] >= 18) & (df["age"] <= 30), "age_group"] = "Young Adult"
    df.loc[(df["age"] >= 31) & (df["age"] <= 60), "age_group"] = "Adult"
    df.loc[df["age"] > 60, "age_group"] = "Senior"

    df["email_domain"] = df["email"].str.split("@").str[1]
    df["loaded_at"] = pd.Timestamp.utcnow()

    before = len(df)
    df = df.drop_duplicates(subset="email")
    if before != len(df):
        logger.warning("Duplicates removed")

    before = len(df)
    df = df.dropna(subset=["email"])
    if before != len(df):
        logger.warning("Rows with missing email removed")

    logger.info(f"Transformed rows: {len(df)}")
    return df


# ---------------- LOAD ----------------
def load(df, db_path="users.db"):
    logger.info("Starting load")

    conn = sqlite3.connect(db_path)
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

    df = df.rename(columns={"nat": "nationality"})

    df["dob_date"] = df["dob_date"].astype(str)
    df["loaded_at"] = df["loaded_at"].astype(str)

    data = list(df[[
        "email","gender","first_name","last_name","nationality",
        "age","age_group","email_domain","dob_date","loaded_at"
    ]].itertuples(index=False, name=None))

    cur.executemany("""
        INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    rows = cur.rowcount
    conn.close()

    logger.info(f"Loaded rows: {rows}")
    return rows


# ---------------- CONTROL ----------------
def get_existing_emails(db_path="users.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT email FROM users")
    emails = {row[0] for row in cur.fetchall()}
    conn.close()
    return emails


def get_last_email(db_path="users.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS etl_control (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cur.execute("SELECT value FROM etl_control WHERE key='last_email'")
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def update_last_email(email, db_path="users.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO etl_control VALUES ('last_email', ?)
    """, (email,))

    conn.commit()
    conn.close()


# ---------------- PIPELINE ----------------
def run_etl():
    try:
        init_db()
        logger.info("ETL started")

        existing = get_existing_emails()

        df_raw = extract(existing)
        df_clean = transform(df_raw)

        if df_clean.empty:
            logger.info("No new data")
            return

        load(df_clean)

        max_email = df_clean["email"].max()
        update_last_email(max_email)

        logger.info("ETL finished")

    except Exception as e:
        with open("alert.log", "a") as f:
            f.write(f"{datetime.utcnow()} ERROR: {str(e)}\n")

        logger.error("Pipeline failed", exc_info=True)
        raise