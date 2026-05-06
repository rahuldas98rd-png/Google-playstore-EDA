import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import numpy as np
from utils.logger import get_logger

logger = get_logger()

class ModelTrainer:
    def __init__(self, df, model_dir="artifacts/models"):
        self.df = df
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        import mlflow
        logger.info("Training models")

        df = self.df.copy()

        df = df.dropna()

        # Features (simplified)
        X = df[['Reviews', 'Installs']]
        y_reg = df['Rating']

        # Success classification
        df['Success'] = (df['Installs'] > 100000).astype(int)
        y_clf = df['Success']

        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2)

        reg_model = RandomForestRegressor()
        reg_model.fit(X_train, y_train)

        clf_model = RandomForestClassifier()
        clf_model.fit(X, y_clf)

        # Regression evaluation
        y_pred = reg_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        logger.info(f"RMSE: {rmse}")

        # Classification evaluation
        y_pred_clf = clf_model.predict(X)
        acc = accuracy_score(y_clf, y_pred_clf)

        logger.info(f"Accuracy: {acc}")

        cm = confusion_matrix(y_clf, y_pred_clf)
        print(cm)

        joblib.dump(reg_model, f"{self.model_dir}/rating_model.pkl")
        joblib.dump(clf_model, f"{self.model_dir}/success_model.pkl")

        with mlflow.start_run():
            mlflow.log_param("model", "RandomForest")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(reg_model, "reg_model")
            mlflow.sklearn.log_model(clf_model, "clf_model")

        logger.info("Models saved successfully")