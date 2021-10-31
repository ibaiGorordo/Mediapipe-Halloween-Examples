# References: 
# - Original model: https://google.github.io/mediapipe/solutions/pose.html
# - 

import cv2
from utils.skeleton_pose_utils import SkeletonPose

show_webcam = True

# Initialize ExorcistFace class
draw_skeleton = SkeletonPose(show_webcam)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Skeleton pose", cv2.WINDOW_NORMAL)

while cap.isOpened():

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    ret, skeleton_image = draw_skeleton(frame)

    if not ret:
        continue

    cv2.imshow("Exorcist face", skeleton_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break