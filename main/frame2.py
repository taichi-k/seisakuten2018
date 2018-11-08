# coding: utf-8
import cv2
import numpy as np
import sys
import time

# 比較元の画像
faces = ["surprised", "smile", "sad", "crying", "laugh", "smug", "nice", "wink"]
detected_counts = [0,0,0,0,0,0,0,0]

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# カメラ
cap = cv2.VideoCapture(int(sys.argv[1]))
cap_w = cap.get(3)
cap_h = cap.get(4)
# cap.set(3, cap_w)
# cap.set(4,cap_h)
# cap.set(5, 30)

#windowsize
window_width = int(sys.argv[2])
window_height = int(sys.argv[2])

emo = cv2.cvtColor(cv2.resize(cv2.imread("img2/1124.jpg"),(400,400),cv2.INTER_LINEAR),cv2.COLOR_BGR2HSV)
fltr = (cv2.resize(np.load('fltr.npy')*255,(400,400),cv2.INTER_LINEAR)/255)[:,:,np.newaxis]
HSV = None

img3 = cv2.resize(cv2.imread("img2/1124.jpg"),(400,400),cv2.INTER_LINEAR)

# def display_face(ura_count,img2):
#     global HSV
#     ura_count = min([ura_count*0.1,1])
#     HSV[:,:,1:] = HSV[:,:,1:]*(1-fltr*ura_count) + emo[:,:,1:]*fltr*ura_count
#     img2 = cv2.cvtColor(HSV,cv2.COLOR_HSV2BGR)
#     return img2

def display_face(ura_count, img2):
    if ura_count > 2:
        global img3
        ura_count *= 0.05
        img2[:,:,:] = img2[:,:,:]*(1-ura_count) + img3[:,:,:] * ura_count

def check_(input_frame):
    global HSV
    hsv = cv2.cvtColor(input_frame,cv2.COLOR_BGR2HSV)
    idx = ((hsv[:,:,0]>20)*(hsv[:,:,0]<30)*(hsv[:,:,1]>90))
    HSV = hsv.copy()
    # input_frame[idx,:] = 255
    if idx.sum() > 60000:
        return True
    # cv2.imshow("f",frame)

def detect(face_name, img2):
    global detected
    global detected_counts
    global faces
    img1 = cv2.imread("img2/" + face_name + ".jpeg")
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    # BFM 生成
    bf = cv2.BFMatcher()
    # 特徴量ベクトル同士を BFM＆KNN でマッチング
    if des2 is not None and des2.shape[0] > 1:
        matches = bf.knnMatch(des1, des2, k=2)
        # 全部だと激重になるのでデータを間引きする
        ratio = 0.6
        good = []
        for data in matches:
            if len(data) > 1:
                m = data[0]
                n = data[1]
                if m.distance < ratio * n.distance:
                    good.append([m])
    # img3 = cv2.drawMatchesKnn(img1, kp1, frame, kp2, good, None, flags=2)
        if len(good) > 3:
            print(face_name+" Existing!!!!!")
            detected = face_name
            detected_counts[faces.index(face_name)] += 1
            return True
        else: return False

# detect
def main():
    global detected
    global detected_counts
    detected = "nothing"
    count = 0
    ura_count = 0
    time_before = time.time()
    found_flg = False
    while(cap.isOpened()):
        # print(detected_counts)
        if detected == "nothing":
            detected = "nothing"
        ret, frame = cap.read()
        found_flg = False
        if frame is not None:
            cv2.rectangle(frame, (int(cap_w/2 - window_width/2), int(cap_h/2 - window_height/2)), (int(cap_w/2 + window_width/2), int(cap_h/2 + window_height/2)), (0,0,255), thickness = 2)
            img2 = frame[int(cap_h/2 - window_height/2):int(cap_h/2 + window_height/2),int(cap_w/2 - window_width/2):int(cap_w/2 + window_width/2)]
            # 特徴量の検出と特徴量ベクトルの計算
            time_after = time.time()
            if time_after - time_before > 0.5:
                time_before = time.time()
                for face in faces:
                    if found_flg == False:
                        found_flg = detect(face, img2)
                    else:
                        break
                if found_flg != True:
                    detected = "nothing"
                    if check_(img2):
                        ura_count = min(ura_count + 2, 12)
                        print("Ura!?!?!?!?")
                    else:
                        ura_count = max(0, ura_count - 3)
                else:
                    ura_count = max(0, ura_count - 3)


            display_face(ura_count,img2)

            if ura_count > 3:
                cv2.putText(frame, "Ura!!!!!", (100, int(cap_h) - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)

            print(ura_count)

            cv2.putText(frame, detected, (int(cap_w) - 300, int(cap_h) - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# idx = np.logical_not((hsv[:,:,0]>20)*(hsv[:,:,0]<30)*(hsv[:,:,1]>90))
# frame[idx,:] = 255
