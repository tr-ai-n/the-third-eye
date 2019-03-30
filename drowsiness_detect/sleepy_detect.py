# import the necessary packages
import numpy as np
import playsound

import imutils
import time
import dlib
import cv2

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread


alarm_path = 'alarm_sound.wav' # alarm sound file path

def sound_alarm(path=alarm_path):
    '''function to play the alarm sound'''
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    
    # two sets of vertical landmark distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # horizontal landmarks distaces
    C = dist.euclidean(eye[0], eye[3])

    # eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# drowsy EYE-Aspect-Ratio threshhold
EAR_THRESH = 0.3

# drowsy frame duration threshold
EAR_FRAMES_THRESH = 45
 
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# face detector (HOG-based)
print("[DEBUG] initializing face detector...")
detector = dlib.get_frontal_face_detector()

# landmark predictor
print("[DEBUG] initializing facial landmark predictor...")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[DEBUG] starting video...")
vs = VideoStream(0).start()
time.sleep(1.0)
 
# read frames from video stream
while True:
    
    # read frame
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check condition if ear is below thresh 
        if ear < EAR_THRESH:

            # COUNTER counts frames
            COUNTER += 1

            # check if COUNTER overrides
            #  EAR_CONSEC_FRAMES threshold
            if COUNTER >= EAR_FRAMES_THRESH:
                # turn ALARM flag ON
                if not ALARM_ON:
                    ALARM_ON = True

                    # create Thread to SOUND ALARM
                    t = Thread(target=sound_alarm, args=(alarm_path,))
                    t.deamon = True
                    t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # If Driver awake, Reset COUNTER and
        # set ALARM_ON flag OFF
        else:
            COUNTER = 0
            ALARM_ON = False

        # drawing EAR on screen for debugging
        print('[DEBUG] EAR value: {:.2f}'.format(ear))

    # display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# closing windows
cv2.destroyAllWindows()
vs.stop()