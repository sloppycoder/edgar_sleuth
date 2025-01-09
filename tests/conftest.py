import os
from pathlib import Path

import psycopg
import pytest

import config

cache_path = str(Path(__file__).parent / "data/cache")
config.setv("cache_path", cache_path)


@pytest.fixture(scope="session")
def clean_db():
    if "_test" not in config.database_url:
        print("Not a test database, skipping cleanup")
        return

    try:
        # Connect to the database
        with psycopg.connect(config.database_url) as conn:
            with conn.cursor() as cur:
                # Fetch all table names in the 'public' schema
                cur.execute("""
                    SELECT tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname = 'public';
                """)
                tables = cur.fetchall()

                # Drop each table
                for table in tables:
                    table_name = table[0]
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")  # pyright: ignore
                conn.commit()
    except Exception as e:
        print(f"Error: {e}")


test_database_url = os.environ.get("TEST_DATABASE_URL")
if test_database_url:
    config.setv("database_url", test_database_url)
