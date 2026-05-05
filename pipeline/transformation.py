from utils.logger import get_logger

logger = get_logger()

class DataTransformation:
    def __init__(self, df):
        self.df = df

    def transform(self):
        logger.info("Transformation started")

        df = self.df.copy()

        # Example: create new feature
        df['Rating_Category'] = df['Rating'].apply(
            lambda x: 'High' if x >= 4 else 'Low'
        )

        logger.info("Transformation completed")
        return df