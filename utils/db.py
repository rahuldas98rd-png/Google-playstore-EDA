# import sqlite3

# DB_PATH = "db/database.db"

# def get_connection():
#     return sqlite3.connect(DB_PATH)

# def save_dataframe(df, table_name="apps"):
#     conn = get_connection()
#     df.to_sql(table_name, conn, if_exists='replace', index=False)
#     conn.close()


from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_URI = os.getenv("DB_URI")

engine = create_engine(DB_URI)

def save_dataframe(df, table_name="apps"):
    try:
        engine = create_engine(DB_URI)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print("✅ Data saved to PostgreSQL")
    except Exception as e:
        print(f"⚠️ DB failed: {e}")