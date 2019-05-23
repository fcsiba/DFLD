# DFLD (DRIVER'S FATIGUE LEVEL DETECTION)
FYP Fall 2018:

Road accidents are suspected to be a primary concern nowadays and the key cause is the fatigue level of the driver which is a subject of intense research today. However, effective driver fatigue monitoring in environments with little or no light has been thought-provoking since the state of the driver is hard to identify.

The system will track driver’s facial features like fast blinking, eye closures, head poses and yawning. This system works at day and at night as well under illuminated conditions through night configurations.

Classifiers like Haar Cascade Classifiers or HOG + linear SVM will be chosen depending on their accuracy and shortest time of computation. EAR (eye aspect ratio) and facial expressions will be used to sense the tiredness of the car driver which will stimulate the system to ring alarms and send warnings consequently. The technical facets of the project will rely on OpenCV and python for facial recognition with Raspberry pi for hardware port connections.

FILES SUBMITTED ARE AS FOLLOWS : 
dfld.py - main code file
alarm.wav - alarm file for eye closure
SOUNDS.wav - sound file for head tilts and yawn
shape_predictor.dat - model file that is used in dfld.py for detecting faces and landmarks.
