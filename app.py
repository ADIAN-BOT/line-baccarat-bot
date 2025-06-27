import os
import uuid
import random
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from flask import Flask, request, abort, send_file
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage,
    RichMenu, RichMenuSize, RichMenuArea, RichMenuBounds,
    URIAction, MessageAction
)
from supabase import create_client, Client
from tensorflow.keras.models import load_model

# === Supabase 設定 ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === LINE 設定 ===
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

@app.route("/trend.png")
def trend_png():
    buffer = generate_trend_chart()
    return send_file(buffer, mimetype='image/png')

# === 使用者資料處理 ===
def get_or_create_user(line_user_id):
    res = supabase.table("members").select("*").eq("line_user_id", line_user_id).execute()
    if res.data:
        return res.data[0]
    else:
        user_code = str(uuid.uuid4())
        new_user = {
            "line_user_id": line_user_id,
            "user_code": user_code,
            "is_authorized": False
        }
        supabase.table("members").insert(new_user).execute()
        return new_user

# === 圖像辨識分析走勢圖（簡化為紅=莊，藍=閒） ===
def analyze_roadmap_image(img_path):
    img = cv2.imread(img_path)
    result_seq = []
    circles = []

    # 設定 HSV 顏色範圍擷取顏色圓形
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))

    # 合併紅色遮罩
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    def detect_centers(mask, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    circles.append((cx, cy, label))

    detect_centers(red_mask, "莊")
    detect_centers(blue_mask, "閒")
    detect_centers(green_mask, "和")

    # 依照 x 排序（橫向時間序）
    circles.sort(key=lambda x: (x[0], x[1]))
    result_seq = [c[2] for c in circles]

    # 轉成數值並輸入模型
    sequence = [1 if r == "莊" else 0 for r in result_seq if r in ["莊", "閒"]][-10:]
    if len(sequence) < 10:
        return 50.0, 50.0, "無法分析（資料不足）"

    model = load_model("baccarat_lstm_model.h5")
    X = np.array(sequence).reshape((1, 10, 1))
    pred = model.predict(X)[0][0]
    banker_rate = round(pred * 100, 1)
    player_rate = round((1 - pred) * 100, 1)
    recommend = "莊" if pred >= 0.5 else "閒"
    return banker_rate, player_rate, recommend

# === 處理圖片訊息（會員上傳走勢圖） ===
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    img_path = f"/tmp/{message_id}.jpg"
    content = line_bot_api.get_message_content(message_id)
    with open(img_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    banker_rate, player_rate, recommend = analyze_roadmap_image(img_path)
    reply = (
        f"📸 圖像分析結果：\n\n"
        f"🔴 莊：{banker_rate}%\n"
        f"🔵 閒：{player_rate}%\n\n"
        f"📈 預測下一顆建議下注：『{recommend}』"
    )
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

