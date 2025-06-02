import os
import joblib
import pandas as pd
import numpy as np
import json

FEATURES = [
    'USMER', 'SEX', 'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA',
    'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL'
]
# Load model and scaler from the model directory
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    print(f"🔍 Loading model from: {model_path}")
    model = joblib.load(model_path)
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"🔍 Loaded scaler from: {scaler_path}")
    return {"model": model, "scaler": scaler}

# Parse incoming request data (JSON only)
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        # Nếu là 1 sample dạng dict, chuyển thành list
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        # Đảm bảo đúng thứ tự và chỉ lấy các feature cần thiết
        df = df[FEATURES]
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    X = input_data.copy()

    # Chuẩn hóa cột 'AGE' nếu scaler tồn tại
    if scaler is not None and 'AGE' in X.columns:
        X.loc[:, 'AGE'] = scaler.transform(X[['AGE']]).flatten()

    predictions = model.predict(X)

    try:
        proba = model.predict_proba(X)
        proba = proba.tolist()
    except Exception as e:
        print(e)
        proba = None

    label_mapping = {0: "LOW", 1: "HIGH"}
    mapped_predictions = [label_mapping[int(pred)] for pred in predictions]

    return {
        "predictions": mapped_predictions,
        "probabilities": proba
    }


# Convert predictions to JSON for response
def output_fn(prediction_dict, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction_dict), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
