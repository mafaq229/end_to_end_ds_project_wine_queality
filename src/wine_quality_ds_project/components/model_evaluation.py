import os
import joblib
from urllib.parse import urlparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

from src.wine_quality_ds_project.config.configuration import ModelEvaluationConfig
from src.wine_quality_ds_project.utils.common import save_json


# os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/muhammadafaq1999/end_to_end_ds_project_wine_queality.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"]="muhammadafaq1999"
# os.environ["MLFLOW_TRACKING_PASSWORD"]="f9459e38c5201df23246f68d9d48551a399fd490"


class ModelEvaluation:
    def __init__(self, config = ModelEvaluationConfig):
        self.config = config
        
    def evaluate_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_registry_uri()).scheme
        
        with mlflow.start_run():
            predited_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.evaluate_metrics(test_y, predited_qualities)
            
            # Saving metrics as local
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")