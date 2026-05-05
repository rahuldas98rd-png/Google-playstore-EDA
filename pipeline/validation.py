from utils.logger import get_logger

logger = get_logger()

class DataValidation:
    def __init__(self, df):
        self.df = df

    def validate(self):
        logger.info("Starting validation")

        if self.df.empty:
            raise ValueError("Dataset is empty")

        required_columns = ['App', 'Category', 'Rating']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

        logger.info("Validation successful")
        return True