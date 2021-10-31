import cv2
import mediapipe as mp
import numpy as np
from imread_from_url import imread_from_url

class SkeletonPose():

	def __init__(self, show_webcam = True, detection_confidence=0.3):

		self.show_webcam = show_webcam

		self.initialize_model(detection_confidence)

		# Read skeleton images
		self.read_skeleton_images()

	def __call__(self, image):

		return self.detect_and_draw_skeleton(image)

	def detect_and_draw_skeleton(self, image):

		self.estimate_pose(image)

		if not self.detected:
			return False, None

		return True, self.draw_skeleton(image)


	def initialize_model(self, detection_confidence):

		# Inialize face mesh detection (0: default model, 1: landmark image optimized)
		self.pose_estimation = mp.solutions.pose.Pose(min_detection_confidence=detection_confidence,
														min_tracking_confidence=0.5)

	def estimate_pose(self, image):

		self.img_height, self.img_width, _ = image.shape

		# Extract background
		input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		input_image.flags.writeable = False
		pose_landmarks = self.pose_estimation.process(input_image).pose_landmarks

		self.detected = pose_landmarks.landmark
		if not self.detected:
			return
		self.skeleton_keypoints = np.array([[int(min([landmark.x*self.img_width,self.img_width-1])), int(min([landmark.y*self.img_height,self.img_height-1]))] 
							for landmark in pose_landmarks.landmark], dtype=np.float32)
		

	def draw_skeleton(self, img):

		if self.show_webcam:
			output_img = img.copy()
		else:
			output_img = np.zeros((self.img_height,  self.img_width, 3), dtype=np.uint8)

		# Draw all the skeleton parts
		self.draw_skull(output_img, self.skeleton_keypoints[skull_indices,:])
		self.draw_torso(output_img, self.skeleton_keypoints[torso_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[left_upper_arm_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[left_lower_arm_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[right_upper_arm_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[right_lower_arm_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[left_upper_leg_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[left_lower_leg_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[right_upper_leg_indices,:])
		self.draw_bone(output_img, self.skeleton_keypoints[right_lower_leg_indices,:])

		return output_img

	
	def draw_skull(self, image, img_skull_keypoints):

		mouth = (img_skull_keypoints[2,:]+img_skull_keypoints[3,:])/2
		img_skull_keypoints = np.vstack((img_skull_keypoints[:2,:],mouth))

		M = cv2.getAffineTransform(skull_image_coordinates, img_skull_keypoints)
		transformed_skull = cv2.warpAffine(self.skull_image, M, (self.img_width, self.img_height))
		transformed_skull_mask = transformed_skull[:,:,2] != 0
		image[transformed_skull_mask] = transformed_skull[transformed_skull_mask]

	def draw_torso(self, image, img_torso_keypoints):

		middle_hip = (img_torso_keypoints[2,:]+img_torso_keypoints[3,:])/2
		img_torso_keypoints = np.vstack((img_torso_keypoints[:2,:],middle_hip))

		original_torso_keypoints = np.array([[0,0],[self.torso_image.shape[1]-1,0],
								[self.torso_image.shape[1]//2,self.torso_image.shape[0]-1]], dtype=np.float32)

		M = cv2.getAffineTransform(original_torso_keypoints, img_torso_keypoints)
		transformed_torso = cv2.warpAffine(self.torso_image, M, (self.img_width, self.img_height))
		transformed_torso_mask = transformed_torso[:,:,2] != 0
		image[transformed_torso_mask] = transformed_torso[transformed_torso_mask]

	def draw_bone(self, image, bone_keypoints):

		original_bone_keypoints = np.array([[self.bone_image.shape[1]//2,0],[self.bone_image.shape[1]//2,self.bone_image.shape[0]-1]], dtype=np.float32)
		M, ret = cv2.estimateAffinePartial2D(original_bone_keypoints, bone_keypoints)

		transformed_bone = cv2.warpAffine(self.bone_image, M, (self.img_width, self.img_height))
		transformed_bone_mask = transformed_bone[:,:,2] != 0
		image[transformed_bone_mask] = transformed_bone[transformed_bone_mask]

	def read_skeleton_images(self):

		self.bone_image = cv2.imread("images/bone.png")
		self.foot_image = cv2.imread("images/foot.png")
		self.hand_image = cv2.imread("images/hand.png")
		self.torso_image = cv2.imread("images/torso.png")
		self.skull_image = cv2.imread("images/skull.png")




skull_indices = [6, 3, 10, 9]
torso_indices = [12, 11, 24, 23]
left_upper_arm_indices = [12, 14]
left_lower_arm_indices = [14, 16]
left_hand_indices = [16, 18, 20]
right_upper_arm_indices = [11, 13]
right_lower_arm_indices = [13, 15]
right_hand_indices = [15, 19, 17]
left_upper_leg_indices = [24, 26]
left_lower_leg_indices = [26, 28]
left_foot_indices = [28, 32]
right_upper_leg_indices = [23, 25]
right_lower_leg_indices = [25, 27]
right_foot_indices = [27, 31]

skull_image_coordinates = np.array([[192,330],[446,349],[324,673]], dtype=np.float32)

def draw_mouseclick_circle(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(skull_image,(x,y),30,(0,0,255),-1)
		print(f"[{x},{y}]")

def select_skull_landmark_pixels():
	# Order: Left eye, right eye, mouth
	cv2.namedWindow("skeleton", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("skeleton",draw_mouseclick_circle)
	while(1):
		cv2.imshow("skeleton",skull_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':

	# Read pumpkin image
	skull_image = cv2.imread("images/skull.png")
	select_skull_landmark_pixels()
