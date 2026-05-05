import sqlite3

DB_PATH = "db/database.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def save_dataframe(df, table_name="apps"):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()