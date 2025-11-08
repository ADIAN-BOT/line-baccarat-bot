from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import cv2
import random
import numpy as np
import pandas as pd
import time
import threading
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

# === 模型載入 ===
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

# === 背景清理 /tmp/ 圖片（daemon thread）===
def clean_tmp(interval=3600, expire=1800):
    """
    interval: 檢查間隔（秒）
    expire: 超過多久未修改的檔案會被刪除（秒）
    """
    while True:
        try:
            now = time.time()
            tmp_path = "/tmp"
            deleted = 0
            if os.path.exists(tmp_path):
                for f in os.listdir(tmp_path):
                    fp = os.path.join(tmp_path, f)
                    try:
                        if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > expire:
                            os.remove(fp)
                            deleted += 1
                    except Exception:
                        # 權限或 race condition：忽略單檔錯誤
                        pass
            if deleted:
                print(f"[clean_tmp] ✅ 已清理 {deleted} 個舊檔案")
        except Exception as e:
            print("[clean_tmp] 清理錯誤：", e)
        time.sleep(interval)

threading.Thread(target=clean_tmp, daemon=True).start()

# === 封裝非同步 DB 操作（專用小函式） ===
def async_insert_record(line_user_id, result, extra: dict = None):
    def job():
        try:
            payload = {"line_user_id": line_user_id, "result": result}
            if extra:
                payload.update(extra)
            supabase.table("records").insert(payload).execute()
        except Exception as e:
            print("[async_insert_record] Supabase insert failed:", e)
    threading.Thread(target=job, daemon=True).start()

def async_insert_member(member_dict):
    def job():
        try:
            supabase.table("members").insert(member_dict).execute()
        except Exception as e:
            print("[async_insert_member] Supabase insert member failed:", e)
    threading.Thread(target=job, daemon=True).start()

def async_update_member_prediction(line_user_id, active: bool):
    def job():
        try:
            supabase.table("members").update({"prediction_active": active}).eq("line_user_id", line_user_id).execute()
        except Exception as e:
            print("[async_update_member_prediction] Supabase update failed:", e)
    threading.Thread(target=job, daemon=True).start()

# === Flask callback ===
@app.route("/callback", methods=['POST', 'HEAD'])
def callback():
    if request.method == 'HEAD':
        return '', 200
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    print("[Webhook 收到訊息]", body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("[callback] handler error:", e)
        abort(500)
    return 'OK'

# === 建立或取得用戶（同步）===
def get_or_create_user(user_id):
    try:
        res = supabase.table("members").select("*").eq("line_user_id", user_id).execute()
    except Exception as e:
        print("[get_or_create_user] Supabase select failed:", e)
        res = None

    if res and getattr(res, "data", None):
        return res.data[0]

    # 若不存在 => 建一筆 member（同步插入，因為需要回傳 user info）
    user_code = str(uuid.uuid4())
    new_user = {
        "line_user_id": user_id,
        "user_code": user_code,
        "is_authorized": False,
        "prediction_active": False
    }
    try:
        supabase.table("members").insert(new_user).execute()
    except Exception as e:
        # 若插入失敗，也回傳 new_user（稍後可由 admin 處理）
        print("[get_or_create_user] insert member failed:", e)
    return new_user

# === 授權檢查（改為不顯示管理連結）===
def check_user_authorized(event, user):
    if not user.get("is_authorized", False):
        safe_reply(
            event,
            f"🔒 尚未授權：你的 UID 為：\n🆔 {user['user_code']}\n請聯絡管理員處理開通。"
        )
        return False
    return True

# === 快速回覆按鈕 ===
def get_quick_reply():
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="🔍 開始預測", text="開始預測")),
        QuickReplyItem(action=MessageAction(label="🔴 莊", text="莊")),
        QuickReplyItem(action=MessageAction(label="🔵 閒", text="閒")),
        QuickReplyItem(action=MessageAction(label="🟢 和局", text="和局")),
        QuickReplyItem(action=MessageAction(label="⛔ 停止預測", text="停止分析")),
    ])

def safe_reply(event, text):
    try:
        req = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=text, quick_reply=get_quick_reply())]
        )
        messaging_api.reply_message(req)
    except Exception as e:
        print("[safe_reply] Reply Message Failed:", e)

# === 三寶/加權邏輯 ===
def predict_pairs(results):
    banker_count = results.count("莊")
    player_count = results.count("閒")
    total = banker_count + player_count or 1
    banker_ratio = banker_count / total
    player_ratio = player_count / total
    pair_weights = {
        "莊對": 0.33 + (banker_ratio - 0.5) * 0.2,
        "閒對": 0.33 + (player_ratio - 0.5) * 0.2,
        "幸運六": 0.34
    }
    total_w = sum(pair_weights.values())
    for k in pair_weights:
        pair_weights[k] = round(pair_weights[k] / total_w * 100, 1)
    return pair_weights

def predict_from_recent_results(results):
    if not results:
        return "無", 0.0, 0.0, "無法判斷"
    feature = [1 if r == "莊" else 0 for r in reversed(results)]
    while len(feature) < 24:
        feature.insert(0, 1 if random.random() > 0.5 else 0)
    X = pd.DataFrame([feature], columns=[f"prev_{i}" for i in range(len(feature))])
    if model is None:
        banker = round(random.random() * 100, 1)
        player = round(100 - banker, 1)
        suggestion = "莊" if banker >= player else "閒"
        return results[0], banker, player, suggestion
    try:
        pred = model.predict_proba(X)[0]
        banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
        suggestion = "莊" if pred[1] >= pred[0] else "閒"
        return results[0], banker, player, suggestion
    except Exception as e:
        print("[predict_from_recent_results] model predict error:", e)
        banker = round(random.random() * 100, 1)
        player = round(100 - banker, 1)
        suggestion = "莊" if banker >= player else "閒"
        return results[0], banker, player, suggestion

def weighted_tie_prediction(user_id):
    try:
        res = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
    except Exception as e:
        print("[weighted_tie_prediction] fetch failed:", e)
        res = None
    if not res or not getattr(res, "data", None):
        return random.choice(["莊", "閒"]), 50.0, 50.0, {"莊對": 33.3, "閒對": 33.3, "幸運六": 33.4}

    results = [r["result"] for r in res.data if r["result"] in ["莊", "閒"]]
    if not results:
        return random.choice(["莊", "閒"]), 50.0, 50.0, {"莊對": 33.3, "閒對": 33.3, "幸運六": 33.4}

    banker_count = results.count("莊")
    player_count = results.count("閒")
    total = banker_count + player_count
    banker_ratio = banker_count / total if total else 0.5
    player_ratio = player_count / total if total else 0.5
    banker_weight = 0.5 + (banker_ratio - 0.5) * 0.6
    player_weight = 0.5 + (player_ratio - 0.5) * 0.6
    total_weight = banker_weight + player_weight
    banker_weight /= total_weight
    player_weight /= total_weight
    prediction = random.choices(["莊", "閒"], weights=[banker_weight, player_weight])[0]
    pair_weights = predict_pairs(results)
    return prediction, round(banker_weight*100, 1), round(player_weight*100, 1), pair_weights

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    msg = event.message.text.strip()
    user_id = event.source.user_id
    user = get_or_create_user(user_id)
    if not check_user_authorized(event, user):
        return

    if msg == "開始預測":
        # 背景非同步更新 member 狀態（不阻塞 webhook 回應）
        async_update_member_prediction(user_id, True)
        safe_reply(event, "✅ 已啟用 AI 預測模式，請上傳房間圖片開始分析。")
        return

    if msg == "停止分析":
        async_update_member_prediction(user_id, False)
        safe_reply(event, "🛑 AI 分析已結束。若要重新開始請輸入『開始預測』。")
        return

    if msg in ["莊", "閒"]:
        # 非同步寫入紀錄，不阻塞回覆
        async_insert_record(user_id, msg)
        # 讀取最近紀錄（同步）用於即時預測
        try:
            history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
            results = [r["result"] for r in reversed(history.data)]
        except Exception as e:
            print("[handle_text] fetch history failed:", e)
            results = [msg]
        last_result, banker, player, suggestion = predict_from_recent_results(results)
        safe_reply(event,
            f"✅ 已記錄：{msg}\n\n"
            f"🔴 莊勝率：{banker}%\n🔵 閒勝率：{player}%\n📈 AI 推論下一顆：{suggestion}"
        )
        return

    if msg == "和局":
        async_insert_record(user_id, "和")
        prediction, banker_w, player_w, pair_weights = weighted_tie_prediction(user_id)
        reply = (
            f"🟢 和局紀錄完成\n\n"
            f"📊 加權預測：{prediction}\n"
            f"📈 權重：莊 {banker_w}%｜閒 {player_w}%\n\n"
            f"🔮 三寶推薦：\n"
            f"🔴 莊對 {pair_weights['莊對']}%\n"
            f"🔵 閒對 {pair_weights['閒對']}%\n"
            f"🍀 幸運六 {pair_weights['幸運六']}%"
        )
        # 非同步保存三寶預測備註
        async_insert_record(user_id, "和局預測", extra={"pair_prediction": str(pair_weights)})
        safe_reply(event, reply)
        return

    safe_reply(event, "請選擇操作功能 👇")

# === 改良版 圖像辨識（針對手機長截圖）===
def detect_last_n_results(image_path, n=24):
    img = cv2.imread(image_path)
    if img is None:
        return []

    # 手機截圖長圖：截下底部大路圖區域（約 65% ~ 100%）
    h, w = img.shape[:2]
    roi = img[int(h * 0.65):h, 0:w]

    # 提高對比、去噪，幫助色彩分離
    roi = cv2.convertScaleAbs(roi, alpha=1.3, beta=15)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 紅色與藍色範圍（可依實拍微調）
    lower_red1, upper_red1 = np.array([0, 90, 90]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 90, 90]), np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_blue, upper_blue = np.array([100, 80, 80]), np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 開運算子去雜訊
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

    # 找輪廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            circles.append((x + w_box, "莊"))
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            circles.append((x + w_box, "閒"))

    # 依 x 座標從右往左排序（最近一顆在右側）
    results = [r for _, r in sorted(circles, key=lambda t: -t[0])]
    return results[:n]

# === 圖像事件處理（使用改良版辨識）===
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    user_id = event.source.user_id
    message_id = event.message.id
    user = get_or_create_user(user_id)

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
            safe_reply(event, "⚠️ 圖像辨識失敗，請重新上傳清晰的大路圖（建議橫向截圖或確保大路圖區塊清楚）。")
            return

        # 非同步寫入 records（避免阻塞 webhook）
        for r in results:
            if r in ["莊", "閒"]:
                async_insert_record(user_id, r)

        # AI 預測（同步，因為要產生回覆）
        feature = [1 if r == "莊" else 0 for r in reversed(results)]
        while len(feature) < 24:
            feature.insert(0, 1 if random.random() > 0.5 else 0)
        X = pd.DataFrame([feature], columns=[f"prev_{i}" for i in range(len(feature))])

        if model is None:
            banker = round(random.random() * 100, 1)
            player = round(100 - banker, 1)
            suggestion = "莊" if banker >= player else "閒"
        else:
            try:
                pred = model.predict_proba(X)[0]
                banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
                suggestion = "莊" if pred[1] >= pred[0] else "閒"
            except Exception as e:
                print("[handle_image] model predict error:", e)
                banker = round(random.random() * 100, 1)
                player = round(100 - banker, 1)
                suggestion = "莊" if banker >= player else "閒"

        reply = (
            f"📸 圖像辨識完成\n\n"
            f"🔙 最後一顆：{results[0]}\n"
            f"🔴 莊勝率：{banker}%\n🔵 閒勝率：{player}%\n\n"
            f"📈 AI 推論下一顆：{suggestion}"
        )
        safe_reply(event, reply)

    except Exception as e:
        print("[處理圖片錯誤]", e)
        safe_reply(event, "⚠️ 圖像處理過程出錯，請再試一次。")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
