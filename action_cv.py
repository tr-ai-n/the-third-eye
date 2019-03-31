from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time
import random
from scipy.misc import imresize
from keras.models import load_model
from datetime import datetime
import pandas as pd

def action_detection():
    audio_files = ["audio/aud5.mp3", "audio/aud6.mp3", "audio/aud5.mp3"]

    class alarm:
        rnd = random.randint(0,2)
        def run(self,path):
            '''function to play the alarm sound'''
            playsound.playsound(path[rnd])

    Alarm = alarm()

    COUNT_TIME_THRESHOLD = 75

    # loading model
    print('[DEBUG] Loading model...')
    model = load_model('distracted_live_model_pose_phone_002.hdf5')

    # start the video stream thread
    print("[DEBUG] starting video stream thread...")
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    # class dictionary
    CLASS_DICT = {  0: 'good_pose',
                    1: 'mobile_phone'}
    # target size
    TARGET_SIZE = (252, 336, 3)

    action = []
    ls = []

    details = {}

    count = 0
    COUNT_TIMER = 0
    # read frames from video strem
    while True:

        # read frame
        ret, frame = cap.read()

        # reshaping the image for MODEL input
        dnn_img = imresize(frame, (216, 384, 3))
        dnn_img = dnn_img.reshape(1, 216, 384, 3)

        # resizing image
        frame = imresize(frame, TARGET_SIZE)
        
        
        # predicting model
        
        #print('DEBUG - 0')
        pred = model.predict(dnn_img).argmax()

        pred = CLASS_DICT[pred]

        if pred=='mobile_phone':
            COUNT_TIMER+=1
            if COUNT_TIMER>COUNT_TIME_THRESHOLD:
                alarm.run(audio_files)
                COUNT_TIMER=0
        else:
            COUNT_TIMER=0

            
        if pred in action:
            ls.append(pd.to_datetime(datetime.now()))
        else:
            
            if len(ls)>1:
                if count not in details:
                    details[count] = {}
                details[count]['start_time'] = ls[0].strftime(dt_ft)
                details[count]['end_time'] = ls[len(ls)-1].strftime(dt_ft)
                details[count]['type_of_action'] = pred
                diff = (ls[len(ls)-1]-ls[0]).seconds
                score = score - (1500/(1500-(25*diff)))
                details[count]['score']=score

                ls = []
                action = []
                count+=1
            else:
                action.append(pred)
                ls.append(pd.to_datetime(datetime.now()))

            


        
        # print('Predicted Class:', pred_class)

        # cv2.putText(frame, pred_class, (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # # display frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
    
    # closing windows
    cap.release()
    cv2.destroyAllWindows()