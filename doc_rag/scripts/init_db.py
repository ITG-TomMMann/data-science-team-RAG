# scripts/init_db.py
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from doc_rag.database.database import Base, engine
from alembic.config import Config
from alembic import command

def init_db():
    """Initialize database tables and run migrations."""
    # Create tables directly (for development only)
    Base.metadata.create_all(bind=engine)
    
    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()