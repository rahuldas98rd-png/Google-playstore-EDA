import pandas as pd
from utils.logger import get_logger

logger = get_logger()

class DataCleaning:
    def __init__(self, df):
        self.df = df

    def clean(self):
        logger.info("Cleaning started")

        df = self.df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

        # Convert Reviews to numeric
        df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

        # Clean Installs
        df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
        df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

        logger.info("Cleaning completed")
        return df