import joblib
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))
    
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction
    
