import os
import uuid
import io
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage,
    RichMenu, RichMenuSize, RichMenuArea, RichMenuBounds,
    URIAction, MessageAction
)
from supabase import create_client, Client
from prediction_model import analyze_image_and_predict

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

# === Rich Menu 建立 ===
def setup_rich_menu():
    rich_menu = RichMenu(
        size=RichMenuSize(width=2500, height=1686),
        selected=False,
        name="Baccarat Menu",
        chat_bar_text="選擇功能",
        areas=[
            RichMenuArea(bounds=RichMenuBounds(0, 0, 833, 843), action=MessageAction(text="開始預測")),
            RichMenuArea(bounds=RichMenuBounds(833, 0, 833, 843), action=MessageAction(text="上閒")),
            RichMenuArea(bounds=RichMenuBounds(1666, 0, 834, 843), action=MessageAction(text="上莊")),
            RichMenuArea(bounds=RichMenuBounds(0, 843, 1250, 843), action=MessageAction(text="使用規則")),
            RichMenuArea(bounds=RichMenuBounds(1250, 843, 1250, 843), action=URIAction(uri="https://wek001.welove777.com"))
        ]
    )
    rich_menu_id = line_bot_api.create_rich_menu(rich_menu)
    with open("richmenu_baccarat.png", 'rb') as f:
        line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", f)
    line_bot_api.set_default_rich_menu(rich_menu_id)
    print("Rich menu created and set:", rich_menu_id)

# === LINE Message 處理 ===
@handler.add(MessageEvent)
def handle_message(event):
    line_user_id = event.source.user_id
    user = get_or_create_user(line_user_id)

    if not user["is_authorized"]:
        reply = (
            f"\U0001F512 此功能僅限授權使用\n\n"
            f"請將下列 UID 複製給管理員進行開通：\n\n"
            f"\U0001F194 {user['user_code']}\n\n"
            f"\U0001F4F2 聯絡管理員：https://lin.ee/2ODINSW"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        return

    if isinstance(event.message, TextMessage):
        msg = event.message.text.strip()
        if msg == "開始預測":
            reply = "✅ 已啟動預測系統，請點選『上莊』或『上閒』，並上傳圖片以獲得下一顆預測"
        elif msg == "上莊" or msg == "上閒":
            last = "莊" if "莊" in msg else "閒"
            supabase.table("records").insert({"line_user_id": line_user_id, "result": last}).execute()
            reply = f"✅ 已紀錄上一顆為『{last}』，請上傳大路圖圖片進行下一顆預測"
        elif msg == "使用規則":
            reply = (
                "📘 使用規則：\n"
                "1. 授權用戶方可使用預測功能\n"
                "2. 每日建議查看最新預測與趨勢\n"
                "3. 請遵守資金管理原則\n"
                "4. 本功能僅供娛樂用途"
            )
        else:
            reply = "請從下方選單選擇操作項目"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif isinstance(event.message, ImageMessage):
        message_id = event.message.id
        img_path = f"/tmp/{message_id}.jpg"
        content = line_bot_api.get_message_content(message_id)
        with open(img_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        last_result, banker_rate, player_rate, predict = analyze_image_and_predict(img_path, supabase)
        reply = (
            f"📸 已判斷上一顆為：{last_result}\n"
            f"🔴 預測莊：{banker_rate}%\n"
            f"🔵 預測閒：{player_rate}%\n\n"
            f"📈 建議下一顆下注：『{predict}』"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
