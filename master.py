# coding: utf-8
import cv2
import numpy as np
import sys, os
import time, math

#list of expression names
# faces = ["surprised", "smile", "sad", "crying", "laugh", "smug"]
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

#width and height for red box
window_width = int(sys.argv[2])
window_height = int(sys.argv[2])

img3 = None
now_str = None

# def display_face(ura_count,img2):
#     global HSV
#     ura_count = min([ura_count*0.1,1])
#     HSV[:,:,1:] = HSV[:,:,1:]*(1-fltr*ura_count) + emo[:,:,1:]*fltr*ura_count
#     img2 = cv2.cvtColor(HSV,cv2.COLOR_HSV2BGR)
#     return img2

# function for taking a picture of our own expressions
def my_capture():
    global now_str
    cap = cv2.VideoCapture(int(sys.argv[1]))
    cap_w = cap.get(3)
    cap_h = cap.get(4)
    # cap.set(3, cap_w)
    # cap.set(4,cap_h)
    # cap.set(5, 30)

    #windowsize
    window_width = int(sys.argv[2])
    window_height = int(sys.argv[2])
    i = 0
    c_faces = ["smile", "sad", "cry", "wink", "angry"]
    now = time.ctime()
    cnvtime = time.strptime(now)
    now_str = (time.strftime("%Y-%m-%d_%H-%M-%S", cnvtime))
    os.mkdir("output/" + now_str)
    flg = False
    sec = 0
    #時間間隔, 何秒ごとに撮影するか
    gap = 5

    while(cap.isOpened()):
        ret, c_frame = cap.read()
        cv2.rectangle(c_frame, (int(cap_w/2 - window_width/2), int(cap_h/2 - window_height/2)), (int(cap_w/2 + window_width/2), int(cap_h/2 + window_height/2)), (0,0,255), thickness = 2)

        # flg to start shooting
        if flg:
            sec = math.floor((time.time() - start))
        else:
            cv2.putText(c_frame, "Press s to start.", (int(cap_w/4), int(cap_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), thickness=4)

        cv2.putText(c_frame, c_faces[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), thickness=2)
        cv2.putText(c_frame, "Countdown :"+str((i+1)*5 - sec), (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=2)
        # 画像表示
        cv2.imshow('frame', c_frame)
        key_pressed = cv2.waitKey(1)
        # if key_pressed & 0xFF == ord('c'):
        #5秒たったら
        if sec == (i+1)*5:
            cv2.imwrite("output/" + now_str + "/" + c_faces[i] + ".jpg", c_frame[int(cap_h/2 - window_height/2):int(cap_h/2 + window_height/2),int(cap_w/2 - window_width/2):int(cap_w/2 + window_width/2)])
            # cv2.imwrite("/Volumes/斉藤のパブリックフォルダ/output/" + "happy" + "_" + str(int(time.time())).replace(".","") + ".jpg", c_frame[int(cap_h/2 - window_height/2):int(cap_h/2 + window_height/2),int(cap_w/2 - window_width/2):int(cap_w/2 + window_width/2)])
            i += 1
            if i == len(c_faces): break
        if key_pressed & 0xFF == ord('q'):
            break
        elif key_pressed & 0xFF == ord("s"):
            start = time.time()
            flg = True

    # cap.release()
    # cv2.destroyAllWindows()


def display_face(ura_count, img2):
    global now_str
    if ura_count > 2:
        global img3
        img3 = cv2.resize(cv2.imread("output/" + now_str + "/" + "smile" + ".jpg"),(400,400),cv2.INTER_LINEAR)
        ura_count *= 0.1
        img2[:,:,:] = img2[:,:,:]*(1-ura_count) + img3[:,:,:] * ura_count

def check_(input_frame):
    global HSV
    hsv = cv2.cvtColor(input_frame,cv2.COLOR_BGR2HSV)
    # Filtering pixels with Yellow Hue. (20~30)
    idx = ((hsv[:,:,0]>20)*(hsv[:,:,0]<30)*(hsv[:,:,1]>90))
    HSV = hsv.copy()
    #here needs tuning of the threshold
    if idx.sum() > 60000:
        return True
    # cv2.imshow("f",frame)

# function for detecting expression of Emoji
# Using AKAZE algorithm
def detect(face_name, img2):
    global detected
    global detected_counts
    global faces
    # reading template imgs
    img1 = cv2.imread("img2/" + face_name + ".jpeg")
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    # BFM 生成
    bf = cv2.BFMatcher()
    # 特徴量ベクトル同士を BFM＆KNN でマッチング
    if des2 is not None and des2.shape[0] > 1:
        matches = bf.knnMatch(des1, des2, k=2)
        # 全部だと激重になるのでデータを間引きする
        # ratioが低いほど厳しい判定になる
        ratio = 0.6
        good = []
        for data in matches:
            if len(data) > 1:
                m = data[0]
                n = data[1]
                if m.distance < ratio * n.distance:
                    good.append([m])

        #もし特徴点が3つ以上あったら
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
    #デフォルトではnothing を表示しておく
    detected = "nothing"
    count = 0
    ura_count = 0
    time_before = time.time()
    found_flg = False
    #何秒ごとにdetect関数を呼ぶか
    interval = 0.5
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
            #interval秒経ったら
            if time_after - time_before > interval:
                time_before = time.time()
                for face in faces:
                    if found_flg == False:
                        found_flg = detect(face, img2)
                    else:
                        break
                if found_flg != True:
                    detected = "nothing"
                    if check_(img2):
                        #ura_count is for counting the number of continuous backface of Emoji detection
                        ura_count = min(ura_count + 2, 7)
                        print("Ura!?!?!?!?")
                    else:
                        ura_count = max(0, ura_count - 3)
                else:
                    ura_count = max(0, ura_count - 3)


            display_face(ura_count,img2)

            if ura_count > 3:
                cv2.putText(frame, "Ura!!!!!", (100, int(cap_h) - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)

            cv2.putText(frame, detected, (int(cap_w) - 300, int(cap_h) - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    my_capture()
    main()

# hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# idx = np.logical_not((hsv[:,:,0]>20)*(hsv[:,:,0]<30)*(hsv[:,:,1]>90))
# frame[idx,:] = 255
