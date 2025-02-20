"""Complete Data Science pipeline for the spaceflights tutorial"""

from .pipeline import create_pipeline  # NOQA
from dotenv import load_dotenv
import os

load_dotenv()

print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
print(f"MLFLOW_SERVER: {os.getenv('MLFLOW_SERVER')}")
