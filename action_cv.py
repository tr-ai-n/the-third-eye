from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time
from scipy.misc import imresize
from keras.models import load_model

# loading model
print('[DEBUG] Loading model...')
model = load_model('models/distracted_live_model_002.hdf5')

# start the video stream thread
print("[DEBUG] starting video stream thread...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# class dictionary
CLASS_DICT = {0: 'good_pose',
 1: 'look_back',
 2: 'phone_call',
 3: 'text_mobile'}
# target size
TARGET_SIZE = (252, 336, 3)

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
    pred_class = CLASS_DICT[pred]
    print('Predicted Class:', pred_class)

    cv2.putText(frame, pred_class, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    # display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# closing windows
cap.release()
cv2.destroyAllWindows()