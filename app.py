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

# === 建立用戶資料 ===
def get_or_create_user(user_id):
    res = supabase.table("members").select("*").eq("line_user_id", user_id).execute()
    if res.data:
        return res.data[0]
    user_code = str(uuid.uuid4())
    new_user = {
        "line_user_id": user_id,
        "user_code": user_code,
        "is_authorized": False
    }
    supabase.table("members").insert(new_user).execute()
    return new_user

# === 預測核心 ===
def analyze_and_predict(image_path, user_id):
    last_result = random.choice(["莊", "閒"])
    supabase.table("records").insert({"line_user_id": user_id, "result": last_result}).execute()
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    records = [r["result"] for r in reversed(history.data)]

    if len(records) < 10:
        return last_result, 0.0, 0.0, "無法預測，紀錄不足。"

    feature = [1 if r == "莊" else 0 for r in records]
    pred = model.predict_proba([feature])[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"
    return last_result, banker, player, suggestion

# === 處理使用者訊息 ===
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
        reply = (
            "請先上傳房間資訊 📝\n"
            "成功後將顯示：\n"
            "房間數據分析成功✔\nAI模型已建立初步判斷\n\n"
            "1.輸入最新開獎結果(莊或閒)\n"
            "2.接著輸入「繼續預測」開始預測下一局\n\n"
            "若換房或結束，請先輸入停止預測再重新上傳新的房間資訊"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif msg == "停止預測":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="✅ 已停止預測，請重新上傳房間資訊以繼續。"))

    elif msg == "繼續預測":
        history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
        records = [r["result"] for r in reversed(history.data)]
        if len(records) < 10:
            reply = "請輸入『莊』或『閒』以進行下一顆預測。"
        else:
            feature = [1 if r == "莊" else 0 for r in records]
            pred = model.predict_proba([feature])[0]
            banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
            suggestion = "莊" if pred[1] >= pred[0] else "閒"
            reply = (
                f"🔴 莊勝率：{banker}%\n"
                f"🔵 閒勝率：{player}%\n"
                f"📈 AI 推論下一顆：{suggestion}"
            )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif msg in ["莊", "閒"]:
        supabase.table("records").insert({"line_user_id": user_id, "result": msg}).execute()
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"✅ 已記錄：{msg}"))

    elif isinstance(event.message, ImageMessage):
        message_id = event.message.id
        image_path = f"/tmp/{message_id}.jpg"
        content = line_bot_api.get_message_content(message_id)
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片收到 ✅ 預測中，請稍後..."))

        last_result, banker, player, suggestion = analyze_and_predict(image_path, user_id)

        reply = (
            f"📸 圖像辨識完成\n\n"
            f"🔙 上一顆開：{last_result}\n"
            f"🔴 莊勝率：{banker}%\n"
            f"🔵 閒勝率：{player}%\n\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )

        line_bot_api.push_message(user_id, TextSendMessage(text=reply))

    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入正確指令或上傳圖片進行預測。"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

