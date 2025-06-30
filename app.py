import os
import uuid
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    RichMenu, RichMenuSize, RichMenuArea, RichMenuBounds,
    URIAction, MessageAction, ImageMessage
)
from supabase import create_client, Client
from prediction_model import analyze_and_predict, model

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
    print("Rich menu created and set:", rich_menu_id)

# === LINE Message 處理 ===
@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
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
            reply = "✅ 已啟動預測系統，請上傳最新路圖圖片"
        elif msg == "使用規則":
            reply = (
                "📘 使用規則：\n"
                "1. 授權用戶方可使用預測功能\n"
                "2. 每日建議查看最新預測與趨勢\n"
                "3. 請遵守資金管理原則\n"
                "4. 本功能僅供娛樂用途"
            )
        elif msg in ["莊", "閒"]:
            supabase.table("records").insert({"line_user_id": line_user_id, "result": msg}).execute()
            reply = f"✅ 已紀錄：{msg}"
        else:
            reply = "請輸入正確指令或上傳圖片進行預測"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

    elif isinstance(event.message, ImageMessage):
        message_id = event.message.id
        img_path = f"/tmp/{message_id}.jpg"
        content = line_bot_api.get_message_content(message_id)
        with open(img_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片收到 ✅ 預測中，請稍後..."))

        banker, player, suggestion = analyze_and_predict(img_path, line_user_id)

        # 新增紀錄進 DB
        supabase.table("records").insert({"line_user_id": line_user_id, "result": suggestion}).execute()

        # 查詢上一顆
        latest = supabase.table("records").select("result").eq("line_user_id", line_user_id).order("id", desc=True).limit(2).execute()
        last_result = latest.data[1]['result'] if len(latest.data) >= 2 else "無紀錄"

        # 查詢最近10顆做預測
        recent = supabase.table("records").select("result").eq("line_user_id", line_user_id).order("id", desc=True).limit(10).execute()
        records = [r["result"] for r in reversed(recent.data)]

        if len(records) < 10:
            pred_msg = "預測樣本不足，請先上傳更多路圖或輸入莊/閒"
        else:
            feature = [1 if r == "莊" else 0 for r in records]
            pred = model.predict_proba([feature])[0]
            b, p = round(pred[1]*100, 1), round(pred[0]*100, 1)
            recommend = "莊" if pred[1] >= pred[0] else "閒"
            pred_msg = (
                f"🔙 上一顆開：{last_result}\n"
                f"🔴 莊勝率：{b}%\n"
                f"🔵 閒勝率：{p}%\n"
                f"📈 AI 推論下一顆：{recommend}"
            )

        line_bot_api.push_message(line_user_id, TextSendMessage(text=pred_msg))

if __name__ == "__main__":
    # setup_rich_menu()  # 首次部署開啟
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

