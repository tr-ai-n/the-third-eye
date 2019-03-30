from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time
from scipy.misc import imresize
# start the video stream thread
print("[DEBUG] starting video stream thread...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# read frames from video strem
while True:

    # read frame
    ret, frame = cap.read()

    # sleep for 3s
    time.sleep(3.0)
    
    # display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# closing windows
cap.release()
cv2.destroyAllWindows()