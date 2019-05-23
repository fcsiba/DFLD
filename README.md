# DFLD (DRIVER'S FATIGUE LEVEL DETECTION)
FYP Fall 2018:

Road accidents are suspected to be a primary concern nowadays and the key cause is the fatigue level of the driver which is a subject of intense research today. However, effective driver fatigue monitoring in environments with little or no light has been thought-provoking since the state of the driver is hard to identify.

The system will track driver’s facial features like fast blinking, eye closures, head poses and yawning. This system works at day and at night as well under illuminated conditions through night configurations.

We used HOG based Classifier. EAR (eye aspect ratio) and facial expressions are used to sense the tiredness of the car driver which will stimulate the system to ring alarms and send warnings consequently. The technical facets of the project will rely on OpenCV and python for facial recognition with Raspberry pi for hardware port connections.

FILES SUBMITTED ARE AS FOLLOWS : 
1) dfld.py - main code file
2) alarm.wav - alarm file for eye closure
3) SOUNDS.wav - sound file for head tilts and yawn
4) shape_predictor.dat - model file that is used in dfld.py for detecting faces and landmarks.

Other relevant documents are uploaded as well.
