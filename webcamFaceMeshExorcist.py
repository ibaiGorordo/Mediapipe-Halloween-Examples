# References: 
# - Original model: https://google.github.io/mediapipe/solutions/face_mesh.html
# - Face swap example: https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python

import cv2
from utils.face_mesh_utils import ExorcistFace

show_webcam = True
max_people = 1

# Image to swap face with
exorcist_image_url = "https://static.wikia.nocookie.net/villains/images/6/66/Theexorcisgirl.png/revision/latest?cb=20190623020548"

# Initialize ExorcistFace class
draw_exorcist = ExorcistFace(exorcist_image_url, show_webcam, max_people)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Exorcist face", cv2.WINDOW_NORMAL)

while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	if not ret:
		continue

	# Flip the image horizontally
	frame = cv2.flip(frame, 1)

	ret, exorcist_image = draw_exorcist(frame)

	if not ret:
		continue
		
	cv2.imshow("Exorcist face", exorcist_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break