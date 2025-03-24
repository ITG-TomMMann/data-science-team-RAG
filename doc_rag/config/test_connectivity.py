import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables from .env file
load_dotenv()

# Retrieve Elasticsearch connection settings
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "localhost:9200")
ELASTIC_USER = os.getenv("ELASTIC_USER", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "")

# Instantiate the Elasticsearch client
es = Elasticsearch(
    ELASTIC_HOST,
    http_auth=(ELASTIC_USER, ELASTIC_PASSWORD)
)

# Test the connection using the ping method
try:
    if es.ping():
        print("Successfully connected to Elasticsearch!")
    else:
        print("Connection to Elasticsearch failed. Check if the server is running and the credentials are correct.")
except Exception as e:
    print("Error connecting to Elasticsearch:", e)
