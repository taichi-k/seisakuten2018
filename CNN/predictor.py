# coding: utf-8
import cv2
import numpy as np
import sys
from keras.models import load_model

#２つモデルを使いアンサンブル
model = load_model('model5.h5')
model2 = load_model('model6.h5')

emo = ['smile','laugh','wink','sad','crying','surprised','smug','null','none']

cap = cv2.VideoCapture(0)
cap_w = cap.get(3)
cap_h = cap.get(4)

#CNNのinput size
size = 64

window_width = 300
window_height = 300

start = (int(cap_w/2)-150,int(cap_h/2)-150)
end = (int(cap_w/2)+150,int(cap_h/2)+150)

def main():
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            cv2.rectangle(frame, start, end, color=(0,0,255), thickness = 2)
            img = frame[start[1]:end[1],start[0]:end[0]]
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame[:,::-1,:])

            ###############################
            if count==10:
                p = -1
                #CNNのinput sizeにresize
                img = cv2.resize(img,(size,size),cv2.INTER_LINEAR)
                #前処理
                img = (img-np.mean(img,axis=(0,1)))/np.std(img,axis=(0,1))
                #アンサンブル
                pred = (model.predict(img[np.newaxis,:,:,:])+model2.predict(img[np.newaxis,:,:,:]))/2
                #予測最大値
                M = np.max(pred)
                #最大値が0.8以上のときのみ予測
                if M>0.8:
                    p = np.argmax(pred)
                print('prediction: %s, %.3f'%(emo[p],M))
                count = 0
            ###############################

        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
