import cv2
import numpy as np
import os
import joblib
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 載入模型
MODEL_PATH = "baccarat_model.pkl"
model = joblib.load(MODEL_PATH)

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
    sequence = [r[2] for r in results if r[2] in ["莊", "閒"]][-10:]

    for r in sequence:
        supabase.table("records").insert({"line_user_id": line_user_id, "result": r}).execute()

    if len(sequence) < 10:
        return 50.0, 50.0, "無法分析（資料不足）"

    feature = [1 if r == "莊" else 0 for r in sequence]
    input_data = np.array(feature).reshape(1, -1)
    pred_proba = model.predict_proba(input_data)[0]
    banker_rate = round(pred_proba[1] * 100, 1)
    player_rate = round(pred_proba[0] * 100, 1)
    recommend = "莊" if pred_proba[1] >= pred_proba[0] else "閒"

    return banker_rate, player_rate, recommend


