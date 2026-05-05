from utils.logger import get_logger
import numpy as np
import pandas as pd

logger = get_logger()

class DataTransformation:
    def __init__(self, df):
        self.df = df

    def transform(self):
        logger.info("Transformation started")

        df = self.df.copy()

        # Log transforms (VERY IMPORTANT)
        df['Log_Reviews'] = np.log1p(df['Reviews'])
        df['Log_Installs'] = np.log1p(df['Installs'])

        # Size cleaning (if exists)
        if 'Size' in df.columns:
            df['Size'] = df['Size'].replace('Varies with device', None)
            df['Size'] = df['Size'].str.replace('M', '')
            df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

        # Encode category
        df = pd.get_dummies(df, columns=['Category'], drop_first=True)

        # Example: create new feature
        df['Rating_Category'] = df['Rating'].apply(
            lambda x: 'High' if x >= 4 else 'Low'
        )

        logger.info("Transformation completed")
        return df