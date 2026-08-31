import os
import google.auth
from google.oauth2 import service_account
from google import genai

DATABASE_SERVICE_URL: str = os.getenv("DATABASE_SERVICE_URL", "http://database_api:8000").rstrip("/")
EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding_service:8001").rstrip("/")

PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
KEY_PATH: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

MODEL_NAME: str = "gemini-2.5-flash-lite"
SCOPES: list[str] = ["https://www.googleapis.com/auth/cloud-platform"]

if os.path.exists(KEY_PATH):
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
else:
    credentials, _ = google.auth.default(scopes=SCOPES)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials
)