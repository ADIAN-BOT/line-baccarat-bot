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
    TextMessage, QuickReply, QuickReplyItem, MessageAction, ReplyMessageRequest,
    PushMessageRequest, Configuration, ApiClient, MessagingApi, MessagingApiBlob
)
from linebot.v3.exceptions import InvalidSignatureError

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
ADMIN_LINE_ID = os.getenv("ADMIN_LINE_ID") # 從環境變數讀取管理員 ID

handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
blob_api = MessagingApiBlob(api_client)

# === Flask App ===
app = Flask(__name__)

# === 🛠️ 管理員通知邏輯 (帶按鈕) ===
def notify_admin_new_user(user_code):
    """
    發送帶有快速回覆按鈕的通知給管理員
    """
    if not ADMIN_LINE_ID:
        print("⚠️ 未設定 ADMIN_LINE_ID，無法發送管理通知")
        return

    # 建立管理員專用的審核按鈕
    admin_quick_reply = QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="✅ 核准授權", text=f"#核准_{user_code}")),
        QuickReplyItem(action=MessageAction(label="❌ 拒絕/關閉", text=f"#取消_{user_code}"))
    ])

    try:
        push_msg = PushMessageRequest(
            to=ADMIN_LINE_ID,
            messages=[TextMessage(
                text=f"🆕 偵測到新用戶申請！\n🆔 UID: {user_code}\n\n請點擊下方按鈕進行審核：",
                quick_reply=admin_quick_reply
            )]
        )
        messaging_api.push_message(push_msg)
    except Exception as e:
        print(f"❌ 管理員 Push 通知失敗: {e}")

# === 背景清理 /tmp/ 圖片（daemon thread）===
def clean_tmp(interval=3600, expire=1800):
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
                        pass
            if deleted:
                print(f"[clean_tmp] ✅ 已清理 {deleted} 個舊檔案")
        except Exception as e:
            print("[clean_tmp] 清理錯誤：", e)
        time.sleep(interval)

threading.Thread(target=clean_tmp, daemon=True).start()

# === 封裝非同步 DB 操作 ===
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
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("[callback] handler error:", e)
        abort(500)
    return 'OK'

# === 🛡️ 建立或取得用戶（修正防護版）===
def get_or_create_user(user_id):
    try:
        res = supabase.table("members").select("*").eq("line_user_id", user_id).execute()
        if res and hasattr(res, "data") and len(res.data) > 0:
            return res.data[0]
        
        # 若不存在 => 生成新 UID 並建一筆 member
        user_code = str(uuid.uuid4())
        new_user = {
            "line_user_id": user_id,
            "user_code": user_code,
            "is_authorized": False,
            "prediction_active": False
        }
        try:
            supabase.table("members").insert(new_user).execute()
        except:
            print("⚠️ 插入資料庫失敗")
        return new_user
    except Exception as e:
        print("[get_or_create_user] error:", e)
        # 關鍵：若資料庫連不通，回傳一個安全字典而不是 None
        return {"line_user_id": user_id, "user_code": "系統連線中...", "is_authorized": False}

# === 🛡️ 授權檢查（修正防護版）===
def check_user_authorized(event, user):
    # 防止 user 為 None 導致程式崩潰
    if not user:
        safe_reply(event, "⚠️ 系統連線異常，請稍後再試。")
        return False

    if not user.get("is_authorized", False):
        user_code = user.get('user_code', '未知')
        if user_code == "系統連線中...":
            safe_reply(event, "🌐 資料庫連線不穩，請稍候片刻再按一次「開始預測」。")
        else:
            notify_admin_new_user(user_code)
            safe_reply(
                event,
                f"🔒 尚未授權：你的 UID 為：\n🆔 {user_code}\n已同步通知管理員開通，請稍候。"
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
        feature.insert(0, 0) 
        
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
        return results[0], 50.0, 50.0, "分析錯誤"

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
    
    if banker_weight >= player_weight:
        prediction = "莊"
    else:
        prediction = "閒"
        
    pair_weights = predict_pairs(results)
    return prediction, round(banker_weight*100, 1), round(player_weight*100, 1), pair_weights

# === 處理文字訊息 ===
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    msg = event.message.text.strip()
    user_id = event.source.user_id

    # 🛠️ 管理員審核指令處理
    if (msg.startswith("#核准_") or msg.startswith("#取消_")) and user_id == ADMIN_LINE_ID:
        try:
            target_code = msg.split("_")[1]
            is_auth = msg.startswith("#核准")
            supabase.table("members").update({"is_authorized": is_auth}).eq("user_code", target_code).execute()
            status_text = "已核准開通" if is_auth else "已關閉權限"
            safe_reply(event, f"✅ 管理員操作成功：\n🆔 UID: {target_code}\n📝 狀態：{status_text}")
            return
        except Exception as e:
            safe_reply(event, f"⚠️ 指令執行出錯：{e}")
            return

    user = get_or_create_user(user_id)

    if msg == "開始預測":
        # 先檢查授權
        if not check_user_authorized(event, user):
            return
        
        async_update_member_prediction(user_id, True)
        safe_reply(event, "✅ 已啟用 AI 預測模式，請上傳房間圖片開始分析。")
        return

    # 一般功能授權牆
    if not check_user_authorized(event, user):
        return

    if msg == "停止分析":
        async_update_member_prediction(user_id, False)
        safe_reply(event, "🛑 AI 分析已結束。若要重新開始請輸入『開始預測』。")
        return

    if msg in ["莊", "閒"]:
        async_insert_record(user_id, msg)
        try:
            history = supabase.table("records").select("result").eq("line_user_id", user_id).order("id", desc=True).limit(10).execute()
            results = [r["result"] for r in reversed(history.data)]
        except Exception as e:
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
        async_insert_record(user_id, "和局預測", extra={"pair_prediction": str(pair_weights)})
        safe_reply(event, reply)
        return

    safe_reply(event, "請選擇操作功能 👇")

# === 【V2.1 圖像辨識優化版】 ===
def detect_last_n_results(image_path, n=24, is_long_mobile_screenshot=True):
    img = cv2.imread(image_path)
    if img is None: return []
    h, w = img.shape[:2]

    if is_long_mobile_screenshot:
        y_start, y_end = int(h * 0.75), int(h * 0.95)
        roi = img[y_start:y_end, 0:w]
        MIN_AREA, MAX_AREA = 50, 800
        MAX_Y_LIMIT = roi.shape[0]
    else:
        roi = img[0:h, 0:w]
        MIN_AREA, MAX_AREA = 150, 800
        MAX_Y_LIMIT = int(h * 0.3)

    roi = cv2.convertScaleAbs(roi, alpha=1.4, beta=20)
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (3, 3), 0), cv2.COLOR_BGR2HSV)

    m_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    m_r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(m_r1, m_r2)
    mask_blue = cv2.inRange(hsv, np.array([90, 100, 80]), np.array([130, 255, 255]))

    circles = []
    def filter_cnts(cnts, label):
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                x, y, wb, hb = cv2.boundingRect(c)
                if not is_long_mobile_screenshot and (y + hb//2) > MAX_Y_LIMIT: continue
                if max(wb/hb, hb/wb) < 1.8:
                    circles.append((x + wb//2, label))

    c_r, _ = cv2.findContours(cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_b, _ = cv2.findContours(cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filter_cnts(c_r, "莊")
    filter_cnts(c_b, "閒")

    results = [r for _, r in sorted(circles, key=lambda t: -t[0])]
    return results[:n]

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    user_id = event.source.user_id
    user = get_or_create_user(user_id)
    
    # 這裡也要檢查授權，防止 NoneType 報錯
    if not check_user_authorized(event, user):
        return
        
    if not user.get("prediction_active", False):
        return

    try:
        image_path = f"/tmp/{event.message.id}.jpg"
        content = blob_api.get_message_content(event.message.id)
        with open(image_path, "wb") as f:
            f.write(b"".join(content.iter_content()) if hasattr(content, "iter_content") else content)

        temp_img = cv2.imread(image_path)
        h, w = temp_img.shape[:2]
        results = detect_last_n_results(image_path, is_long_mobile_screenshot=(h/w >= 1.5))
        
        if not results:
            safe_reply(event, "⚠️ 圖像辨識失敗，請重新上傳清晰的大路圖。")
            return

        for r in results:
            if r in ["莊", "閒"]: async_insert_record(user_id, r)

        feature = [1 if r == "莊" else 0 for r in reversed(results)]
        while len(feature) < 24: feature.insert(0, 0)
        X = pd.DataFrame([feature], columns=[f"prev_{i}" for i in range(len(feature))])

        if model is None:
            banker = round(random.random() * 100, 1)
            suggestion = "莊" if banker >= 50 else "閒"
            player = 100 - banker
        else:
            pred = model.predict_proba(X)[0]
            banker, player = round(pred[1]*100, 1), round(pred[0]*100, 1)
            suggestion = "莊" if pred[1] >= pred[0] else "閒"

        safe_reply(event, f"📸 圖像辨識完成\n\n🔙 已記錄走勢\n🔴 莊勝率：{banker}%\n🔵 閒勝率：{player}%\n\n📈 AI 推論下一顆：{suggestion}")
    except Exception as e:
        print("[handle_image] error:", e)
        safe_reply(event, "⚠️ 圖像處理出錯。")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=False)
