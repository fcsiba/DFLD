# DFLD - FYP
import dlib
import cv2
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
from threading import Thread
import playsound
import argparse
import imutils
import time


def play_alarm(path):
    # START ALARM TO WAKE UP THE DRIVER!
    playsound.playsound(path)


def EAR_calculate(eye):
    #FIRST WE ARE COMPUTING EUCLIDEAN DISTANCE BW VERTICAL EYE LANDMARKS i.e  : their x,y coordinates
    v1 = distance.euclidean(eye[1], eye[5])
    v2 = distance.euclidean(eye[2], eye[4])

    # Doing the same for horizontal
    h = distance.euclidean(eye[0], eye[3])

    # Now computing EAR with the formula :
    EAR = (v1 + v2) / (2.0 * h)

    return EAR


# setting threshold and consecutive frames limit
Eye_threshold = 0.3
EYE_CONSEC_FRAMES = 45
LIP_CONSEC_FRAMES = 8
HEAD_TILT_CONSEC_FRAMES = 10

#setting counters
COUNTER_EYE = 0
COUNTER_EYELIP = 0
COUNTER_EYEHEAD_TILT = 0

#boolean for setting alarm on or off
ALARM = False

## sounds for yawn and headtilts
SOUNDS_ON=False
##

# Using Dlib face detector (HOG-based)
print(" DFLD is about to start.. ")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")

# Getting indexes for left and right eye
(lefteyestart, lefteyeend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(righteyestart, righteyeend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

## EXTRACTING MOUTH REGION
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#####

#STARTING VIDEO STREAM THREAD
video_stream = VideoStream(src=0).start()
time.sleep(1.0)

# LOOP for the frames from the video stream

while True:
    # From the threaded video frame is taken and resized
    frame = video_stream.read()
    frame = imutils.resize(frame, width=450)
    #CONVERTING IT INTO GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces in gray scale frame now
    faces = detector(gray, 0)

    # looping over facial detections
    for face in faces:
        # Determining the facial landmarks for the facial region
        # Converting the facial landmark (x, y) coordinates to a NumPy Array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lefteyestart:lefteyeend]
        rightEye = shape[righteyestart:righteyeend]
       #

       #for mouth
        mouth=shape[mStart:mEnd]

       #For top lip
        top_lip_pts = []
        for i in range(50, 53):
            top_lip_pts.append(shape[i])
        for i in range(61, 64):
            top_lip_pts.append(shape[i])
        top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
        top_lip_mean = np.mean(top_lip_pts, axis=0)
        top_lip_mean= int(top_lip_mean[1])

       #

        #For bottom lip
        bottom_lip_pts = []
        for i in range(65, 68):
            bottom_lip_pts.append(shape[i])
        for i in range(56, 59):
            bottom_lip_pts.append(shape[i])
        bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
        bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
        bottom_lip_mean= int(bottom_lip_mean[1])
        lip_distance = abs(top_lip_mean - bottom_lip_mean)
        #print(lip_distance)

        if(lip_distance>25):
            COUNTER_EYELIP += 1
            if COUNTER_EYELIP >= LIP_CONSEC_FRAMES:
              if not SOUNDS_ON:
                SOUNDS_ON = True

                if "SOUNDS.wav" != "":
                    t = Thread(target=play_alarm,
                               args=("SOUNDS.wav",))
                    t.deamon = True
                    t.start()

                    cv2.putText(frame, "YAWN DETECTED!", (30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            COUNTER_EYELIP=0
            SOUNDS_ON = False
                               # HEAD TILTS HERE :
        #getting the angle first
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]

        angle = np.degrees(np.arctan2(dY, dX))
        #COUNTER_EYEHEAD_TILT , HEAD_TILT_CONSEC_FRAMES
        # print(leftEye[0][0])
        # print(rightEye[0][0])
        if((angle >133 and angle <143) or (angle >-143 and angle < -133)):
            COUNTER_EYEHEAD_TILT += 1
            if COUNTER_EYEHEAD_TILT >= HEAD_TILT_CONSEC_FRAMES:
                if not SOUNDS_ON:
                  SOUNDS_ON = True
                  if "SOUNDS.wav" != "":
                    t = Thread(target=play_alarm,
                               args=("SOUNDS.wav",))
                    t.deamon = True
                    t.start()
                    cv2.putText(frame, "HEAD TILT DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            COUNTER_EYEHEAD_TILT = 0
            SOUNDS_ON = False

        leftEAR = EAR_calculate(leftEye)
        rightEAR = EAR_calculate(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #visualize mouth
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1)
        #
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame COUNTER_EYE
        if ear < Eye_threshold:
            COUNTER_EYE += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER_EYE >= EYE_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM:
                    ALARM = True

                    # Verify if alarm file is there
                    # Starting a new thread to have the alarm sound played in the background
                    if "alarm.wav" != "":
                        t = Thread(target=play_alarm,
                                   args=("alarm.wav",))
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "WAKE UP!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the COUNTER_EYE and alarm
        else:
            COUNTER_EYE = 0
            ALARM = False

        #  eye aspect ratio printing it to help in testing.

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_ITALIC, 0.7, (0, 255, 255), 2)

    # display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

#PRESS E TO EXIT AND BREAK 
    if key == ord("e"):
        break

# CLEANING UP ..
cv2.destroyAllWindows()
#stop stream
video_stream.stop()