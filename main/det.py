# coding: utf-8
import cv2
import numpy as np
import sys
# 比較元の画像
img1 = cv2.imread(sys.argv[1])
this_len = int(sys.argv[2])
img1 = cv2.resize(img1, (this_len,this_len))

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# カメラ
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)
cap.set(5, 15)

# detect
def detect():
    while(cap.isOpened()):
        ret, frame = cap.read()

        # 特徴量の検出と特徴量ベクトルの計算
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(frame, None)

        # BFM 生成
        bf = cv2.BFMatcher()

        # 特徴量ベクトル同士を BFM＆KNN でマッチング
        matches = bf.knnMatch(des1, des2, k=2)

        # 全部だと激重になるのでデータを間引きする
        ratio = 0.6
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])

        # 特徴点同士を描画してつなぐ
        img3 = cv2.drawMatchesKnn(img1, kp1, frame, kp2, good, None, flags=2)

        # 画像表示
        cv2.imshow('frame', img3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
