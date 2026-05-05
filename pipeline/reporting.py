import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import get_logger

logger = get_logger()

class ReportGenerator:
    def __init__(self, df, output_dir="artifacts/reports"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self):
        logger.info("Generating EDA reports")

        # Rating distribution
        plt.figure()
        sns.histplot(self.df['Rating'], bins=20)
        plt.title("Rating Distribution")
        plt.savefig(f"{self.output_dir}/rating_dist.png")
        plt.close()

        # Category count
        plt.figure(figsize=(10,5))
        self.df['Category'].value_counts().head(10).plot(kind='bar')
        plt.title("Top Categories")
        plt.savefig(f"{self.output_dir}/category.png")
        plt.close()

        logger.info("Reports generated successfully")