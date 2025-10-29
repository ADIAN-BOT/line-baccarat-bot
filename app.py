from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import cv2
import random
import numpy as np
import pandas as pd
from flask import Flask, request, abort
from supabase import create_client, Client
import joblib

from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import (
    TextMessage, QuickReply, QuickReplyItem, MessageAction, ReplyMessageRequest
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import MessagingApi, MessagingApiBlob, Configuration, ApiClient

# === 載入模型 ===
try:
    model = joblib.load("baccarat_model_trained.pkl")
except Exception as e:
    print("❌ 模型載入失敗：", e)
    model = None

# === 初始化 Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 初始化 LINE ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

handler = WebhookHandler(LINE_CHANNEL_SECRET)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)

messaging_api = MessagingApi(api_client)
blob_api = MessagingApiBlob(api_client)

# === Flask App ===
app = Flask(__name__)

@app.route("/callback", methods=['POST', 'HEAD'])
def callback():
    if request.method == 'HEAD':
        return '', 200
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    print("[Webhook 收到訊息]", body)
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

# === 授權檢查 ===
def check_user_authorized(event, user):
    """未授權者直接回覆提示並中斷流程"""
    if not user.get("is_authorized", False):
        safe_reply(
            event,
            f"🔒 尚未授權，請將以下 UID 提供給管理員開通：\n🆔 {user['user_code']}\n📩 聯絡管理員：https://lin.ee/2ODINSW"
        )
        return False
    return True

# === 圖像分析辨識前 N 顆莊或閒 ===
def detect_last_n_results(image_path, n=24):
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
    red_circles = [(cv2.boundingRect(cnt), '莊') for cnt in contours_red if cv2.contourArea(cnt) > 100]
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_circles = [(cv2.boundingRect(cnt), '閒') for cnt in contours_blue if cv2.contourArea(cnt) > 100]
    all_circles = [(x+w, res) for ((x, y, w, h), res) in red_circles + blue_circles]
    sorted_results = [r for _, r in sorted(all_circles, key=lambda t: -t[0])]
    return sorted_results[:n]

# === 預測邏輯 ===
def predict_from_recent_results(results):
    if not results:
        return "無", 0.0, 0.0, "無法判斷"
    feature = [1 if r == "莊" else 0 for r in reversed(results)]
    while len(feature) < 24:
        feature.insert(0, 1 if random.random() > 0.5 else 0)
    X = pd.DataFrame([feature], columns=[f"prev_{i}" for i in range(len(feature))])
    pred = model.predict_proba(X)[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"
    return results[0], banker, player, suggestion

# === 快速回覆按鈕 ===
def get_quick_reply():
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="🔍 開始預測", text="開始預測")),
        QuickReplyItem(action=MessageAction(label="🔴 莊", text="莊")),
        QuickReplyItem(action=MessageAction(label="🔵 閒", text="閒")),
        QuickReplyItem(action=MessageAction(label="🟢 和局", text="和局")),
        QuickReplyItem(action=MessageAction(label="⛔ 停止預測", text="停止分析")),
    ])

# === 安全回覆 ===
def safe_reply(event, message_text):
    try:
        req = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=message_text, quick_reply=get_quick_reply())]
        )
        messaging_api.reply_message(req)
    except Exception as e:
        print("[Error] Reply Message Failed:", str(e))

# === 和局加權預測 ===
def weighted_tie_prediction(user_id):
    res = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    if not res.data:
        return random.choice(["莊", "閒"]), 50.0, 50.0

    results = [r["result"] for r in res.data if r["result"] in ["莊", "閒"]]
    banker_count = results.count("莊")
    player_count = results.count("閒")
    total = banker_count + player_count
    if total == 0:
        return random.choice(["莊", "閒"]), 50.0, 50.0

    banker_ratio = banker_count / total
    player_ratio = player_count / total
    avg = (banker_ratio + player_ratio) / 2

    banker_weight = 0.5 + (banker_ratio - avg) * 0.6
    player_weight = 0.5 + (player_ratio - avg) * 0.6

    total_weight = banker_weight + player_weight
    banker_weight /= total_weight
    player_weight /= total_weight

    prediction = random.choices(["莊", "閒"], weights=[banker_weight, player_weight])[0]
    return prediction, round(banker_weight * 100, 1), round(player_weight * 100, 1)

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    msg = event.message.text.strip()
    user_id = event.source.user_id
    user = get_or_create_user(user_id)

    # --- 授權檢查 ---
    if not check_user_authorized(event, user):
        return

    # === 預測流程 ===
    if msg == "開始預測":
        supabase.table("members").update({"prediction_active": True}).eq("line_user_id", user_id).execute()
        safe_reply(event, "✅ 已啟用 AI 預測模式，請上傳房間圖片開始分析。")
        return

    if msg == "停止分析":
        supabase.table("members").update({"prediction_active": False}).eq("line_user_id", user_id).execute()
        safe_reply(event, "🛑 AI 分析已結束。若要重新開始請輸入『開始預測』。")
        return

    if msg in ["莊", "閒"]:
        supabase.table("records").insert({"line_user_id": user_id, "result": msg}).execute()
        history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
        results = [r["result"] for r in reversed(history.data)]
        last_result, banker, player, suggestion = predict_from_recent_results(results)
        safe_reply(event, f"✅ 已記錄：{msg}\n\n🔴 莊勝率：{banker}%\n🔵 閒勝率：{player}%\n📈 AI 推論下一顆：{suggestion}")
        return

    if msg == "和局":
        supabase.table("records").insert({"line_user_id": user_id, "result": "和"}).execute()
        weighted_choice, banker_weight, player_weight = weighted_tie_prediction(user_id)
        reply = (
            f"🟢 和局紀錄完成\n\n"
            f"📊 根據最近莊閒比例加權預測：{weighted_choice}\n"
            f"📈 權重：莊 {banker_weight}%｜閒 {player_weight}%"
        )
        safe_reply(event, reply)
        return

    safe_reply(event, "請選擇操作功能 👇")

# === 處理圖片 ===
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    user_id = event.source.user_id
    message_id = event.message.id
    user = get_or_create_user(user_id)

    # --- 授權檢查 ---
    if not check_user_authorized(event, user):
        return

    if not user.get("prediction_active", False):
        safe_reply(event, "⚠️ 請先輸入『開始預測』以啟用分析。")
        return

    try:
        image_path = f"/tmp/{message_id}.jpg"
        content = blob_api.get_message_content(message_id)
        with open(image_path, "wb") as f:
            f.write(content)

        results = detect_last_n_results(image_path)
        if not results:
            safe_reply(event, "⚠️ 圖像辨識失敗，請重新上傳清晰的大路圖。")
            return

        for r in results:
            if r in ["莊", "閒"]:
                supabase.table("records").insert({"line_user_id": user_id, "result": r}).execute()

        feature = [1 if r == "莊" else 0 for r in reversed(results)]
        while len(feature) < 24:
            feature.insert(0, 1 if random.random() > 0.5 else 0)
        X = pd.DataFrame([feature], columns=[f"prev_{i}" for i in range(len(feature))])
        pred = model.predict_proba(X)[0]
        banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
        suggestion = "莊" if pred[1] >= pred[0] else "閒"
        last_result = results[0]

        reply = (
            f"📸 圖像辨識完成\n\n"
            f"🔙 最後一顆：{last_result}\n"
            f"🔴 莊勝率：{banker}%\n"
            f"🔵 閒勝率：{player}%\n\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )
        safe_reply(event, reply)

    except Exception as e:
        print("[處理圖片錯誤]", e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
