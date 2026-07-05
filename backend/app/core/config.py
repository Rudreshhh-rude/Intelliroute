import os
from typing import Optional
from pydantic import BaseModel

def load_dotenv(dotenv_path: str) -> None:
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key:
                        os.environ[key] = val

# Load .env from project root relative to backend/app/core/config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
dotenv_file = os.path.join(root_dir, ".env")
load_dotenv(dotenv_file)

# Fallback to current working directory if not loaded yet
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(".env")
    load_dotenv(os.path.join(os.getcwd(), ".env"))

class Settings(BaseModel):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "backend/chromadb_store")
    model_name_flash: str = os.getenv("MODEL_NAME_FLASH", "gemini-2.5-flash")
    model_name_pro: str = os.getenv("MODEL_NAME_PRO", "gemini-2.5-pro")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

settings = Settings()
