# References:
# - Mediapipe Hair segmentation model (Not supported in Python due to the custom operations): https://google.github.io/mediapipe/solutions/hair_segmentation
# - Model used in this example: https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP
# - Read gif from url: https://stackoverflow.com/questions/48163539/how-to-read-gif-from-url-using-opencv-python

import cv2
import numpy as np

from utils.fire_hair_utils import HairSegmentation, get_fire_gif

# Initialize webcam
cap = cv2.VideoCapture(0)
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Fire Hair", cv2.WINDOW_NORMAL)

# Inialize hair segmentation model
hair_segmentation = HairSegmentation(webcam_width, webcam_height)

while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	img_height, img_width, _ = frame.shape

	if not ret:
		continue

	# Flip the image horizontally
	frame = cv2.flip(frame, 1)

	# Segment hair
	hair_mask = hair_segmentation(frame)

	# Draw fire 
	combined_image = hair_segmentation.draw_fire_hair(frame, hair_mask)

	cv2.imshow("Fire Hair", combined_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
