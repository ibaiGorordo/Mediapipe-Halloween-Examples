# References: 
# - Original model: https://google.github.io/mediapipe/solutions/face_detection.html
# - Pumpkin image: https://pixabay.com/photos/pumpkin-fruit-orange-fall-2805140/

import cv2
import mediapipe as mp
import numpy as np

from utils.pumpkin_face_utils import read_pumpkin_image, draw_pumpkins

show_webcam = False

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pumpkin face", cv2.WINDOW_NORMAL)

# Read pumpkin image
pumpkin_image_path = "https://cdn.pixabay.com/photo/2017/10/01/11/36/pumpkin-2805140_960_720.png"
pumpkin_image = read_pumpkin_image(pumpkin_image_path)

# Inialize background segmentation (0: small model for distace < 2m, 1: full range model for distance < 5m)
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	img_height, img_width, _ = frame.shape

	if not ret:
		continue

	# Flip the image horizontally
	frame = cv2.flip(frame, 1)

	# Detect face
	input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	input_image.flags.writeable = False
	detections = face_detection.process(input_image).detections

	# Draw pumkins
	output_img = draw_pumpkins(frame, pumpkin_image, detections, show_webcam)

	cv2.imshow("Pumpkin face", output_img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

