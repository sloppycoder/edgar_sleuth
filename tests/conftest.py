import os
from pathlib import Path

import pytest

import config
from sleuth.datastore import execute_query

cache_path = str(Path(__file__).parent / "data/cache")
config.setv("cache_path", cache_path)

test_database_url = os.getenv("TEST_DATABASE_URL")
if test_database_url:
    os.environ["DATABASE_URL"] = test_database_url


@pytest.fixture(scope="session")
def clean_db():
    try:
        current_database = execute_query("SELECT current_database() as db")[0]["db"]
        if "_test" not in current_database:
            print("Not a test database, skipping cleanup")
            return

        for table in execute_query("""
                    SELECT tablename FROM pg_catalog.pg_tables
                    WHERE schemaname = 'public';
                """):
            execute_query(f"DROP TABLE IF EXISTS {table["tablename"]} CASCADE;")
    except Exception as e:
        print(f"Error: {e}")
