# direct_db_test.py
from sqlalchemy import create_engine, text

# Hardcoded connection string for testing
DATABASE_URL = "postgresql://admin:pass132@localhost:5432/ragdb"

try:
    # Create engine and test connection
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {str(e)}")