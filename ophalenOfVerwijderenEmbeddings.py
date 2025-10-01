import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

# Kies de modus: "select" om rijen op te vragen, "delete" om alle rijen te verwijderen
MODE = "select"   # verander dit naar "delete" om alle rijen te wissen

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

def main():
    conn = get_connection()
    register_vector(conn)  # nodig om vector kolommen als Python-lijsten te lezen
    cur = conn.cursor()

    try:
        if MODE == "select":
            cur.execute("""
                SELECT id, document, embedding, created_at
                FROM embedding
                ORDER BY created_at DESC
                LIMIT 1;
            """)
            row = cur.fetchone()
            if not row:
                print("Geen rijen gevonden.")
                return

            row_id, document, embedding, created_at = row

            print("ID:", row_id)
            print("Created at:", created_at)
            print("Document (eerste 200 chars):", (document or "")[:200])
            if embedding is None:
                print("Embedding: NULL")
            else:
                print("Embedding dimensie:", len(embedding))
                print("Embedding voorbeeld (eerste 10 waarden):", embedding[:10])

        elif MODE == "delete":
            cur.execute("DELETE FROM embedding;")
            conn.commit()
            print("✅ Alle rijen verwijderd uit tabel embedding.")

        else:
            print(f"Onbekende MODE: {MODE}")

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()