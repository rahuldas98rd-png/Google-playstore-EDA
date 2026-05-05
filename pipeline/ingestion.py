import pandas as pd
from utils.logger import get_logger

logger = get_logger()

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise