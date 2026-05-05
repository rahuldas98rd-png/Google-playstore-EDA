import os
from pipeline.ingestion import DataIngestion
from pipeline.validation import DataValidation
from pipeline.cleaning import DataCleaning
from pipeline.transformation import DataTransformation
from pipeline.reporting import ReportGenerator
from pipeline.model import ModelTrainer
from utils.db import save_dataframe
from utils.logger import get_logger

logger = get_logger()

class DataPipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        logger.info("Pipeline started")

        # Ingestion
        ingestion = DataIngestion(self.input_path)
        df = ingestion.load_data()

        # Validation
        validator = DataValidation(df)
        validator.validate()

        # Cleaning
        cleaner = DataCleaning(df)
        df_clean = cleaner.clean()

        # Transformation
        transformer = DataTransformation(df_clean)
        df_final = transformer.transform()

        # Reports
        reporter = ReportGenerator(df_final)
        reporter.generate()

        # ML Models
        trainer = ModelTrainer(df_final)
        trainer.train()

        # Save artifact
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, "processed.csv")
        df_final.to_csv(output_file, index=False)

        save_dataframe(df_final)

        logger.info(f"Pipeline completed. Saved to {output_file}")
        return df_final