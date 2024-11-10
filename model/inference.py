import json
import xgboost as xgb
import numpy as np
import os
import io
import pandas as pd


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.json")
    model = xgb.Booster()
    model.load_model(model_path)
    
    return model

def input_fn(input_data, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(input_data)
        df = pd.DataFrame(data)
        
        return df
    else:
        raise ValueError(f"Unsupported content type {content_type}")

def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix)
    
    return predictions

def output_fn(predictions, accept="application/json"):
    if accept == "application/json":
        return json.dumps(predictions.tolist()), accept
    else:
        raise ValueError(f"Unsupported accept type {accept}")
