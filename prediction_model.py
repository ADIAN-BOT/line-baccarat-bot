import cv2
import numpy as np
from tensorflow.keras.models import load_model
from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = load_model("baccarat_lstm_model.h5")

def analyze_and_predict(img_path, line_user_id):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, "無法讀取圖片"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 顏色範圍
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

    circles = []

    def find_centers(mask, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    circles.append((cx, cy, label))

    find_centers(red_mask, "莊")
    find_centers(blue_mask, "閒")

    # 依照 X、Y 排序
    circles.sort(key=lambda x: (x[0], x[1]))
    results = [c[2] for c in circles]

    if not results:
        return None, None, "無法偵測任何開獎結果"

    # 記錄最後一顆
    last_result = results[-1]
    supabase.table("records").insert({"line_user_id": line_user_id, "result": last_result}).execute()

    # 建立預測用輸入序列
    seq = [1 if r == "莊" else 0 for r in results if r in ["莊", "閒"]][-10:]
    if len(seq) < 10:
        return None, None, "資料不足，無法預測"

    X = np.array(seq).reshape((1, 10, 1))
    pred = model.predict(X)[0][0]
    banker_rate = round(pred * 100, 1)
    player_rate = round((1 - pred) * 100, 1)
    recommend = "莊" if pred >= 0.5 else "閒"

    return banker_rate, player_rate, recommend
