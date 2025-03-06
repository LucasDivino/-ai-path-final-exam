from dotenv import load_dotenv
import os

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
ENCODER_ID = int(os.getenv("ENCODER_ID", "1"))