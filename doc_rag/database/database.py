# doc_rag/models/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from doc_rag.config.settings import get_settings

settings = get_settings()

# Select connection method based on environment
# Update doc_rag/models/database.py

from doc_rag.config.settings import get_settings
import os

settings = get_settings()

# Determine if using Cloud SQL or standard PostgreSQL connection
if settings.POSTGRES_HOST.startswith("/cloudsql/"):
    # Cloud SQL with Unix socket
    db_socket_dir = os.environ.get("DB_SOCKET_DIR", "/cloudsql")
    instance_connection_name = settings.POSTGRES_HOST.replace("/cloudsql/", "")
    
    # Unix socket connection string
    db_url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@/{settings.POSTGRES_DB}?host={db_socket_dir}/{instance_connection_name}"
else:
    # Standard TCP connection
    db_url = settings.DATABASE_URL

engine = create_engine(db_url)

engine = create_engine(
    db_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections before using from pool
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()