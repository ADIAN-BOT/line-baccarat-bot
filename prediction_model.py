import os
import tempfile
import requests
import cv2
import numpy as np
from flask import Flask, request, abort
from supabase import create_client, Client
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import joblib
import random

# === 載入模型 ===
model = joblib.load("baccarat_model.pkl")

# === 初始化 Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def analyze_and_predict(image_path, user_id):
    # 圖像處理與辨識（這裡可替換為你的 OCR 大路圖解析邏輯）
    # 模擬從圖片預測出『上一顆』結果（真實應改為影像分析）
    last_result = random.choice(["莊", "閒"])

    # 將上一顆結果寫入資料庫
    supabase.table("records").insert({"line_user_id": user_id, "result": last_result}).execute()

    # 取得最近10筆記錄
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records = [r["result"] for r in reversed(history.data)]

    if len(records) < 10:
        return last_result, 0.0, 0.0, "無法預測，紀錄不足。"

    # 特徵轉換與模型預測
    feature = [1 if r == "莊" else 0 for r in records]
    pred = model.predict_proba([feature])[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"

    return last_result, banker, player, suggestion


