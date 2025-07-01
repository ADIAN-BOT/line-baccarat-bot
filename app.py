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
    MessageEvent, TextMessage, TextSendMessage, ImageMessage,
    QuickReply, QuickReplyButton, MessageAction
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
        "prediction_active": False,
        "await_continue": False
    }
    supabase.table("members").insert(new_user).execute()
    return new_user

# === 圖像分析辨識最後一顆莊或閒 ===
def detect_last_result(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_circles = [cv2.boundingRect(cnt) for cnt in contours_red if cv2.contourArea(cnt) > 100]
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_circles = [cv2.boundingRect(cnt) for cnt in contours_blue if cv2.contourArea(cnt) > 100]
    if not red_circles and not blue_circles:
        return None
    all_circles = [(x+w, '莊') for (x, y, w, h) in red_circles] + [(x+w, '閒') for (x, y, w, h) in blue_circles]
    last = sorted(all_circles, key=lambda t: -t[0])[0][1]
    return last

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

# === 快速回覆按鈕 ===
def get_quick_reply():
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="🔍 開始預測", text="開始預測")),
        QuickReplyButton(action=MessageAction(label="🔴 莊", text="莊")),
        QuickReplyButton(action=MessageAction(label="🔵 閒", text="閒")),
        QuickReplyButton(action=MessageAction(label="▶️ 繼續分析", text="繼續分析")),
        QuickReplyButton(action=MessageAction(label="⛔ 停止預測", text="停止分析")),
        QuickReplyButton(action=MessageAction(label="📘 使用說明", text="使用說明")),
        QuickReplyButton(action=MessageAction(label="🔗 註冊網址", text="註冊網址")),
    ])

# === LINE Message 處理 ===
@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event):
    user_id = event.source.user_id
    user = get_or_create_user(user_id)
    msg = event.message.text if isinstance(event.message, TextMessage) else None

    if not user['is_authorized']:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text=(
                "🔒 尚未授權，請將以下 UID 提供給管理員開通：\n"
                f"🆔 {user['user_code']}\n"
                "📩 聯絡管理員：https://lin.ee/2ODINSW"
            ),
            quick_reply=get_quick_reply()
        ))
        return

    if msg == "使用說明":
        usage = (
            "📘 使用說明：\n\n"
            "1️⃣ 開始預測前請先複製 UID 給客服人員\n"
            "2️⃣ 開通後即可開始操作，操作步驟如下：\n"
            "🔹 上傳你所在房間的大路圖表格\n"
            "🔹 圖片分析成功後，會自動回傳上一顆是莊或閒\n"
            "🔹 回傳結果後，請點『繼續分析』再進行下一步預測\n"
            "🔹 換房或結束後，請點『停止分析』關閉分析功能"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=usage, quick_reply=get_quick_reply()))
        return

    if msg == "註冊網址":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="🔗 點擊進入註冊頁面：https://wek001.welove777.com",
            quick_reply=get_quick_reply()
        ))
        return

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請選擇操作功能 👇", quick_reply=get_quick_reply()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

