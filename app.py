import time
import os
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel


PREPROCESS_PATH = "./preprocess"
MODEL_PATH = "./model/WDL.onnx"

def load_resources(dir_path):
    le_dict = joblib.load(os.path.join(dir_path, "le_dict.pkl"))
    scaler  = joblib.load(os.path.join(dir_path, "scaler.pkl"))
    meta    = joblib.load(os.path.join(dir_path, "features_meta.pkl"))
    return le_dict, scaler, meta

def preprocess_input(input_dict, le_dict, scaler, meta):
    sparse_features = meta['sparse_features']
    dense_features = meta['dense_features']

    sparse_vector = []
    for feat in sparse_features:
        val = str(input_dict.get(feat, "-1"))
        encoder = le_dict[feat]
        encoded_val = encoder.transform([val])[0]
        sparse_vector.append(encoded_val)

    dense_vector = []
    for feat in dense_features:
        val = float(input_dict.get(feat, 0.0))
        dense_vector.append(val)

    dense_vector = np.array([dense_vector])
    dense_vector = scaler.transform(dense_vector)

    return (
        np.array([sparse_vector], dtype=np.int64),
        dense_vector.astype(np.float32)
    )
LE_DICT, SCALER, META = load_resources(PREPROCESS_PATH)
SESSION = ort.InferenceSession(MODEL_PATH)

app = FastAPI()

# 요청 Body 정의
class Sample(BaseModel):
    age: int = 0
    gender: str = ""
    region: str = ""
    spending_power: str = ""
    fav_category: str = ""

    ad_category: str = ""
    price_level: str = ""
    target_gender: str = ""
    target_ages: str = ""

    os: str = ""
    network_type: str = ""
    time_hour: int = 0

    fav_match: int = 0
    gender_match: int = 0
    age_match: int = 0
    age_similarity: str = "0.0"
    is_target_match: int = 0


@app.post("/predict")
def predict(sample: Sample):
    s=time.time()
    raw_dict = sample.dict()

    sparse_input, dense_input = preprocess_input(
        raw_dict, LE_DICT, SCALER, META
    )
    
    outputs = SESSION.run(None, {
        "sparse_input": sparse_input,
        "dense_input": dense_input
    })


    ctr_prob = outputs[0][0][0]
    e=time.time()
    return {"ctr_prob": float(ctr_prob), "time": float(e-s)}