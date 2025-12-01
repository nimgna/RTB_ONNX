# test.py
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--age", type=int, required=True)
parser.add_argument("--gender", type=str, required=True)
parser.add_argument("--region", type=str, required=True)
parser.add_argument("--spending_power", type=str, required=True)
parser.add_argument("--fav_category", type=str, required=True)
parser.add_argument("--ad_category", type=str, required=True)
parser.add_argument("--price_level", type=str, required=True)
parser.add_argument("--target_gender", type=str, required=True)
parser.add_argument("--target_ages", type=str, required=True)
parser.add_argument("--os", type=str, required=True)
parser.add_argument("--network_type", type=str, required=True)
parser.add_argument("--time_hour", type=int, required=True)
parser.add_argument("--fav_match", type=int, required=True)
parser.add_argument("--gender_match", type=int, required=True)
parser.add_argument("--age_match", type=int, required=True)
parser.add_argument("--age_similarity", type=str, required=True)
parser.add_argument("--is_target_match", type=int, required=True)

args = parser.parse_args()
data = vars(args)

print(">> 요청 JSON:", data)

res = requests.post("http://localhost:8000/predict", json=data)

print(">> STATUS:", res.status_code)

# 실패했을 때는 raw text 먼저 보여주기
if res.status_code != 200:
    print(">> RAW RESPONSE:", res.text)
else:
    print(">> JSON RESPONSE:", res.json())