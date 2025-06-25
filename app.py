import os
import uuid
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from supabase import create_client, Client

# Supabase 設定（這裡改成正確的名稱）
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# LINE 設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Flask App
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

# 使用者資料處理
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

# 處理 LINE 訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_user_id = event.source.user_id
    msg = event.message.text.strip()
    user = get_or_create_user(line_user_id)
    if not user["is_authorized"]:
        reply = (
            f"🔒 此功能僅限授權使用\n\n"
            f"請將下列 UID 複製給管理員進行開通：\n\n"
            f"🆔 {user['user_code']}\n\n"
            f"📲 聯絡管理員：https://lin.ee/2ODINSW"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        return

    if msg in ["預測", "開始預測"]:
        reply = "📊 預測結果：建議下注『莊』，下一局請小心操作！"
    else:
        reply = "請輸入『預測』以啟動分析系統"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

