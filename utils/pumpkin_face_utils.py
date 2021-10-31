# References:
# - Mouse click coordinates: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture/28330835

import cv2
import numpy as np

from urllib.request import Request, urlopen
from imread_from_url import imread_from_url

# Lanmark coordinates (left eye, right eye, mouth) in the pumpkin image to match with the face landmarks obtained running this script
pumpkin_landmarks = np.array([[347,338],[547,326],[458,548]], dtype=np.float32)

def draw_pumpkins(image, pumpkin_image, detections, show_webcam=False):
	
	pumpkin_image_height, pumpkin_image_width, _ = pumpkin_image.shape

	if show_webcam:
		output_img = cv2.resize(image, (pumpkin_image_width, pumpkin_image_height))
	else:
		output_img = np.zeros((pumpkin_image_height, pumpkin_image_width, 3), dtype=np.uint8)

	if detections:
		for detection in detections:
			face_coordinates = np.array([[detection.location_data.relative_keypoints[i].x*pumpkin_image_width, detection.location_data.relative_keypoints[i].y*pumpkin_image_height] for i in [0,1,3]], dtype=np.float32)
			M = cv2.getAffineTransform(pumpkin_landmarks, face_coordinates)
			transformed_pumpkin = cv2.warpAffine(pumpkin_image, M, (pumpkin_image_width, pumpkin_image_height))
			transformed_pumpkin_mask = transformed_pumpkin[:,:,3] != 0
			output_img[transformed_pumpkin_mask] = transformed_pumpkin[transformed_pumpkin_mask,:3]

	return output_img

def read_pumpkin_image(pumpkin_image_path):
	req = Request(pumpkin_image_path, headers={'User-Agent': 'Mozilla/5.0'})
	arr = np.asarray(bytearray(urlopen(req).read()), dtype=np.uint8)
	return  cv2.imdecode(arr,  cv2.IMREAD_UNCHANGED)

def draw_mouseclick_circle(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(pumpkin_image,(x,y),30,(0,0,255),-1)
		print(f"[{x},{y}]")

def select_pumpkin_landmark_pixels():
	# Order: Left eye, right eye, mouth
	cv2.namedWindow("Pumpkin face", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("Pumpkin face",draw_mouseclick_circle)
	while(1):
		cv2.imshow("Pumpkin face",pumpkin_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':

	# Read pumpkin image
	pumpkin_image_path = "https://cdn.pixabay.com/photo/2017/10/01/11/36/pumpkin-2805140_960_720.png"
	pumpkin_image = imread_from_url(pumpkin_image_path)

	select_pumpkin_landmark_pixels()