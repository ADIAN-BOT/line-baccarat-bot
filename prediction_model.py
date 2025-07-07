import os
import requests
import cv2
import numpy as np
import random
import joblib
from supabase import create_client, Client

# === 載入模型 ===
model = joblib.load("baccarat_model.pkl")

# === 初始化 Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def analyze_and_predict(image_path, user_id):
    # TODO: 替換這段為真正的圖像辨識結果
    last_result = random.choice(["莊", "閒"])

    # 寫入資料庫
    supabase.table("records").insert({"line_user_id": user_id, "result": last_result}).execute()

    # 取得最近 10 筆記錄
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records = [r["result"] for r in reversed(history.data)]

    if len(records) < 10:
        return last_result, 0.0, 0.0, "無法預測，紀錄不足。"

    # 特徵轉換：莊=1, 閒=0
    features = [1 if r == "莊" else 0 for r in records]
    prediction = model.predict_proba([features])[0]

    banker_prob = round(prediction[1] * 100, 1)
    player_prob = round(prediction[0] * 100, 1)
    suggestion = "莊" if prediction[1] >= prediction[0] else "閒"

    return last_result, banker_prob, player_prob, suggestion
