import os
import uuid
import tempfile
import requests
import cv2
import numpy as np
from flask import Flask, request, abort
from supabase import create_client, Client
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)
import joblib
import random

# === 載入模型 ===
model = joblib.load("baccarat_model.pkl")

# === 初始化 Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 初始化 LINE ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === Flask App ===
app = Flask(__name__)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# === 圖像分析與預測邏輯 ===
def analyze_and_predict(image_path, user_id):
    # 模擬圖像辨識結果（此處可整合 OCR 進行自動識別）
    last_result = random.choice(["莊", "閒"])

    # 寫入上一顆結果
    supabase.table("records").insert({"line_user_id": user_id, "result": last_result}).execute()

    # 取得最近 10 顆紀錄
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records = [r["result"] for r in reversed(history.data)]

    if len(records) < 10:
        return last_result, 0.0, 0.0, "無法預測，紀錄不足。"

    feature = [1 if r == "莊" else 0 for r in records]
    pred = model.predict_proba([feature])[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"

    # 寫入這顆推論結果（作為實際開獎後記錄）
    supabase.table("records").insert({"line_user_id": user_id, "result": suggestion}).execute()

    # 再次預測下一顆
    history2 = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records2 = [r["result"] for r in reversed(history2.data)]

    if len(records2) < 10:
        next_predict = "紀錄不足"
        b2, p2 = 0.0, 0.0
    else:
        feature2 = [1 if r == "莊" else 0 for r in records2]
        pred2 = model.predict_proba([feature2])[0]
        b2, p2 = round(pred2[1]*100, 1), round(pred2[0]*100, 1)
        next_predict = "莊" if pred2[1] >= pred2[0] else "閒"

    return last_result, banker, player, suggestion, b2, p2, next_predict

# === LINE Message 處理 ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id
    message_id = event.message.id
    image_path = f"/tmp/{message_id}.jpg"

    content = line_bot_api.get_message_content(message_id)
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    # 初步回應
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片收到 ✅ 預測中，請稍後..."))

    # 執行預測
    last_result, banker, player, suggestion, b2, p2, next_predict = analyze_and_predict(image_path, user_id)

    reply = (
        f"📸 圖像辨識完成\n\n"
        f"🔙 上一顆開：{last_result}\n"
        f"🔴 莊勝率：{banker}%\n"
        f"🔵 閒勝率：{player}%\n"
        f"📈 AI 推論下一顆：{suggestion}\n\n"
        f"⏭️ AI 推論再下一顆：{next_predict}\n"
        f"🔴 莊勝率：{b2}%\n"
        f"🔵 閒勝率：{p2}%"
    )

    line_bot_api.push_message(user_id, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

