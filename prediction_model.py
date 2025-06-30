import cv2
import numpy as np
import os
from supabase import create_client, Client

# === Supabase 連線 ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 顏色圈圈辨識
def detect_circles_by_color(img, lower, upper, label):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 1500:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                results.append((cx, cy, label))
    return results

# 查詢歷史紀錄
def get_recent_results(user_id, limit=5):
    res = supabase.table("records") \
        .select("result") \
        .eq("line_user_id", user_id) \
        .order("id", desc=True) \
        .limit(limit) \
        .execute()
    return [r["result"] for r in reversed(res.data)]

# 規則預測（基於過去5顆統計）
def rule_based_prediction(history):
    if len(history) < 5:
        return 50.0, 50.0, "無法分析（歷史紀錄不足）"

    banker_count = history.count("莊")
    player_count = history.count("閒")
    banker_rate = round(banker_count / 5 * 100, 1)
    player_rate = round(player_count / 5 * 100, 1)
    suggestion = "莊" if banker_rate >= player_rate else "閒"
    return banker_rate, player_rate, suggestion

# 主函數
def analyze_and_predict(img_path, line_user_id):
    img = cv2.imread(img_path)
    if img is None:
        return 50.0, 50.0, "無法分析：圖片讀取失敗"

    red1 = ((0, 70, 50), (10, 255, 255))
    red2 = ((170, 70, 50), (180, 255, 255))
    blue = ((100, 100, 100), (130, 255, 255))

    results = []
    results += detect_circles_by_color(img, *red1, label="莊")
    results += detect_circles_by_color(img, *red2, label="莊")
    results += detect_circles_by_color(img, *blue, label="閒")

    results.sort(key=lambda x: (x[0], x[1]))
    sequence = [r[2] for r in results if r[2] in ["莊", "閒"]]

    # 寫入最新一顆
    if sequence:
        latest = sequence[-1]
        supabase.table("records").insert({"line_user_id": line_user_id, "result": latest}).execute()

    # 抓歷史做預測
    history = get_recent_results(line_user_id)
    return rule_based_prediction(history)

