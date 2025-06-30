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

app = Flask(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        temp_file_path = tf.name

    user_id = event.source.user_id
    banker, player, prediction = analyze_and_predict(temp_file_path, user_id)

    # 取得上一顆
    history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(2).execute()
    last_result = "無紀錄"
    if history.data and len(history.data) >= 2:
        last_result = history.data[1]["result"]

    reply = f"上一顆開：{last_result}\n莊勝率：{banker}%\n閒勝率：{player}%\n建議下注：{prediction}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if text in ["莊", "閒"]:
        supabase.table("records").insert({"line_user_id": user_id, "result": text}).execute()

        # 取得最近 10 筆結果進行預測
        history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
        records = [r["result"] for r in reversed(history.data)]

        if len(records) < 10:
            reply = "無法預測，紀錄不足。"
        else:
            from prediction_model import model
            feature = [1 if r == "莊" else 0 for r in records]
            pred = model.predict_proba([feature])[0]
            banker, player = round(pred[1]*100,1), round(pred[0]*100,1)
            recommend = "莊" if pred[1] >= pred[0] else "閒"
            reply = f"上一顆開：{text}\n莊勝率：{banker}%\n閒勝率：{player}%\n建議下注：{recommend}"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


