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

# === 圖像分析辨識前 N 顆莊或閒 ===
def detect_last_n_results(image_path, n=10):
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

# === 圖像分析與預測邏輯（直接使用偵測結果） ===
def predict_from_recent_results(results):
    if not results:
        return "無", 0.0, 0.0, "無法判斷"
    feature = [1 if r == "莊" else 0 for r in reversed(results)]
    while len(feature) < 10:
        feature.insert(0, 1 if random.random() > 0.5 else 0)
    pred = model.predict_proba([feature])[0]
    banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
    suggestion = "莊" if pred[1] >= pred[0] else "閒"
    return results[0], banker, player, suggestion

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

    if msg == "開始預測":
        supabase.table("members").update({"prediction_active": True, "await_continue": False}).eq("line_user_id", user_id).execute()
        reply = (
            "請先上傳房間資訊 📝\n"
            "成功後將顯示：\n"
            "房間數據分析成功✔\nAI模型已建立初步判斷\n\n"
            "後續每次上傳圖片將自動辨識並進行預測。\n"
            "若換房或結束，請輸入『停止分析』再重新上傳新的房間圖。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply, quick_reply=get_quick_reply()))
        return

    if msg == "停止分析":
        supabase.table("members").update({"prediction_active": False, "await_continue": False}).eq("line_user_id", user_id).execute()
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="🛑 AI 分析已結束，若需進行新的預測請先上傳房間圖片並點擊『開始預測』重新啟用。",
            quick_reply=get_quick_reply()
        ))
        return

    if msg == "繼續分析":
        supabase.table("members").update({"await_continue": False}).eq("line_user_id", user_id).execute()
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="✅ AI 已繼續分析，請輸入『莊』或『閒』以進行下一筆預測。", quick_reply=get_quick_reply()))
        return

    if msg in ["莊", "閒"]:
        if user.get("await_continue", False):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ 請先輸入『繼續分析』以進行下一步預測。", quick_reply=get_quick_reply()))
            return
        supabase.table("records").insert({"line_user_id": user_id, "result": msg}).execute()
        history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
        results = [r["result"] for r in reversed(history.data)]
        last_result, banker, player, suggestion = predict_from_recent_results(results)
        reply = (
            f"✅ 已記錄：{msg}\n\n"
            f"🔴 莊勝率：{banker}%\n"
            f"🔵 閒勝率：{player}%\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply, quick_reply=get_quick_reply()))
        supabase.table("members").update({"await_continue": True}).eq("line_user_id", user_id).execute()
        return

    if isinstance(event.message, ImageMessage):
        if not user.get("prediction_active", False):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="⚠️ 預測尚未啟動，請先輸入『開始預測』以啟用分析。",
                quick_reply=get_quick_reply()
            ))
            return

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片收到 ✅ 預測中，請稍後..."))
        message_id = event.message.id
        image_path = f"/tmp/{message_id}.jpg"
        content = line_bot_api.get_message_content(message_id)
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)
        results = detect_last_n_results(image_path)
        if not results:
            line_bot_api.push_message(user_id, TextSendMessage(text="⚠️ 圖像辨識失敗，請重新上傳清晰的大路圖。", quick_reply=get_quick_reply()))
            return
        for r in results:
            supabase.table("records").insert({"line_user_id": user_id, "result": r}).execute()
        last_result, banker, player, suggestion = predict_from_recent_results(results)
        reply = (
            f"📸 圖像辨識完成\n\n"
            f"🔙 最後一顆：{last_result}\n"
            f"🔴 莊勝率：{banker}%\n"
            f"🔵 閒勝率：{player}%\n\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )
        line_bot_api.push_message(user_id, TextSendMessage(text=reply, quick_reply=get_quick_reply()))
        supabase.table("members").update({"await_continue": True}).eq("line_user_id", user_id).execute()
        return

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請選擇操作功能 👇", quick_reply=get_quick_reply()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

