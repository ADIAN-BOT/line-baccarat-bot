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

# === 建立或取得用戶 ===
def get_or_create_user(user_id):
    res = supabase.table("members").select("*").eq("line_user_id", user_id).execute()
    if res.data:
        return res.data[0]
    user_code = str(uuid.uuid4())
    new_user = {
        "line_user_id": user_id,
        "user_code": user_code,
        "is_authorized": False,
        "prediction_active": False
    }
    supabase.table("members").insert(new_user).execute()
    return new_user

# === 圖像分析辨識莊或閒 ===
def detect_last_result(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    roi = img[h-50:h, w-100:w]  # 擷取右下角格子範圍
    avg_color = cv2.mean(roi)[:3]
    r, g, b = avg_color
    if r > 150 and b < 100:
        return "莊"
    elif b > 150 and r < 100:
        return "閒"
    else:
        return None

# === 圖像分析與預測邏輯 ===
def analyze_and_predict(user_id):
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records = [r["result"] for r in reversed(history.data)]

    if len(records) < 10:
        return "無", 0.0, 0.0, "紀錄不足，請先多上傳幾張圖片建立預測紀錄"

    feature = [1 if r == "莊" else 0 for r in records]
    pred = model.predict_proba([feature])[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"
    last_result = records[-1]
    return last_result, banker, player, suggestion

# === LINE Message 處理 ===
@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event):
    user_id = event.source.user_id
    user = get_or_create_user(user_id)

    if not user['is_authorized']:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=(
                "🔒 尚未授權，請將以下 UID 提供給管理員開通：\n"
                f"🆔 {user['user_code']}\n"
                "📩 聯絡管理員：https://lin.ee/2ODINSW"
            ))
        )
        return

    msg = event.message.text if isinstance(event.message, TextMessage) else None

    if msg == "開始預測":
        supabase.table("members").update({"prediction_active": True}).eq("line_user_id", user_id).execute()
        reply = (
            "請先上傳房間資訊 📝\n"
            "成功後將顯示：\n"
            "房間數據分析成功✔\nAI模型已建立初步判斷\n\n"
            "後續每次上傳圖片將自動辨識並進行預測。\n"
            "若換房或結束，請輸入『停止分析』再重新上傳新的房間圖。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif msg == "停止分析":
        supabase.table("members").update({"prediction_active": False}).eq("line_user_id", user_id).execute()
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="🛑 AI 分析已結束，若需進行新的預測請先上傳房間圖片並點擊『開始預測』重新啟用。"
        ))

    elif msg in ["莊", "閒"]:
        supabase.table("records").insert({"line_user_id": user_id, "result": msg}).execute()
        last_result, banker, player, suggestion = analyze_and_predict(user_id)
        reply = (
            f"✅ 已記錄：{msg}\n\n"
            f"🔴 莊勝率：{banker}%\n"
            f"🔵 閒勝率：{player}%\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif isinstance(event.message, ImageMessage):
        if not user.get("prediction_active", False):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="⚠️ 預測尚未啟動，請先輸入『開始預測』以啟用分析。"
            ))
            return

        message_id = event.message.id
        image_path = f"/tmp/{message_id}.jpg"
        content = line_bot_api.get_message_content(message_id)
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        detected = detect_last_result(image_path)
        if detected in ["莊", "閒"]:
            supabase.table("records").insert({"line_user_id": user_id, "result": detected}).execute()
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ 圖像辨識失敗，請重新上傳清晰的大路圖。"))
            return

        last_result, banker, player, suggestion = analyze_and_predict(user_id)

        if suggestion.startswith("紀錄不足"):
            reply = "📸 圖像辨識完成\n⚠️ AI 無法預測，紀錄不足。"
        else:
            reply = (
                f"📸 圖像辨識完成\n\n"
                f"🔙 上一顆開：{last_result}\n"
                f"🔴 莊勝率：{banker}%\n"
                f"🔵 閒勝率：{player}%\n\n"
                f"📈 AI 推論下一顆：{suggestion}"
            )

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入正確指令或上傳圖片進行預測。"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

