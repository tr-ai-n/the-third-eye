#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import time


# In[2]:


import cv2


# In[5]:

def return_known_name(path):

    CAPTURED_NAME = ''
    video_capture = cv2.VideoCapture(0)

    print('embedding face')
    my_face_image = face_recognition.load_image_file(path)
    my_face_encoding = face_recognition.face_encodings(my_face_image)[0]


    known_face_encodings = [my_face_encoding]

    known_face_names = ['Shashwat']


    face_locations = []
    face_encodings = []

    face_names = []
    process_this_frame = True

    print('starting video...')
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    print('[DEBUG] KNOWN FACE DETECTED: ', name)
                    CAPTURED_NAME = name
                    break
        break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    return CAPTURED_NAME

if __name__=='__main__':
    None