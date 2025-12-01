import os
import joblib
import numpy as np
import onnxruntime as ort


PREPORSS_PATH="./preprocess"
MODEL_PATH = os.path.join("./model", "WDL.onnx") # xxx.onnx

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

raw_sample = {
    # User
    "age": 29,
    "gender": "M",
    "region": "수원시",
    "spending_power": "High",
    "fav_category": "Game",

    # Item
    "ad_category": "Game",
    "price_level": "Mid",
    "target_gender": "Both",
    "target_ages": "['20-30', '30-40']",

    # Context
    "os": "Android",
    "network_type": "5g",
    "time_hour": 15,

    # Cheats
    "fav_match": 0, #취향 일치 여부 0 or 1
    "gender_match": 0, #성별 일치 여부 0 or 1
    "age_match": 0, #연령 타겟팅 일치 여부 0 or 1
    "age_similarity": "0.0", #연령대가 얼마나 유사한지 0~1
    "is_target_match": 0 #타겟팅 완전 일치 여부 0 or 1
}

le_dict, scaler, meta = load_resources(PREPORSS_PATH)
session = ort.InferenceSession(MODEL_PATH)
print("모델 로드 완료")

sparse_input, dense_input = preprocess_input(raw_sample, le_dict, scaler, meta)
print("전처리 완료")

input_feed = {
    "sparse_input": sparse_input,
    "dense_input": dense_input
}

outputs = session.run(None, input_feed)
ctr_prob = outputs[0][0][0]

print(f"클릭 확률 : {ctr_prob}%")