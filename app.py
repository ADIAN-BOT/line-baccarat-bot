import os
import uuid
import random
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    RichMenu, RichMenuSize, RichMenuArea, RichMenuBounds,
    URIAction, MessageAction
)
from supabase import create_client, Client

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
        areas=[
            RichMenuArea(bounds=RichMenuBounds(0, 0, 500, 1686), action=MessageAction(text="開始預測")),
            RichMenuArea(bounds=RichMenuBounds(500, 0, 500, 1686), action=MessageAction(text="莊")),
            RichMenuArea(bounds=RichMenuBounds(1000, 0, 500, 1686), action=MessageAction(text="閒")),
            RichMenuArea(bounds=RichMenuBounds(1500, 0, 500, 1686), action=MessageAction(text="使用規則")),
            RichMenuArea(bounds=RichMenuBounds(2000, 0, 500, 1686), action=URIAction(uri="https://wek001.welove777.com")),
        ]
    )
    rich_menu_id = line_bot_api.create_rich_menu(rich_menu)
    with open("richmenu_baccarat.png", 'rb') as f:
        line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", f)
    line_bot_api.set_default_rich_menu(rich_menu_id)
    print("Rich menu ID:", rich_menu_id)

# === AI 模擬預測 ===
def predict_next_result():
    res = supabase.table("records").select("result").order("created_at", desc=True).limit(50).execute()
    results = [r["result"] for r in res.data if r["result"] in ["莊", "閒"]]

    if not results:
        return 50.0, 50.0, random.choice(["莊", "閒"])

    banker_count = results.count("莊")
    player_count = results.count("閒")
    total = banker_count + player_count or 1

    banker_rate = round((banker_count / total) * 100, 1)
    player_rate = round((player_count / total) * 100, 1)
    recommend = "莊" if banker_rate > player_rate else "閒"

    return banker_rate, player_rate, recommend

# === 處理 LINE 訊息 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_user_id = event.source.user_id
    msg = event.message.text.strip()
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

    if msg == "開始預測":
        reply = "✅ 已啟動預測系統，請選擇『莊』或『閒』"
    elif msg in ["莊", "閒"]:
        banker_rate, player_rate, recommend = predict_next_result()
        reply = (
            f"📊 AI 勝率分析：\n\n"
            f"🔴 莊：{banker_rate}%\n"
            f"🔵 閒：{player_rate}%\n\n"
            f"📈 預測下一顆建議下注：『{recommend}』"
        )
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

if __name__ == "__main__":
    # 初始化 Rich Menu：首次部署請取消註解，之後可關閉避免重複建
    setup_rich_menu()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))


