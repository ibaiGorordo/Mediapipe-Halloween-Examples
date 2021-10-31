# Mediapipe-Halloween-Examples
Python scripts using the Mediapipe models for Halloween.

# WHY
Mainly for fun. But this repository also includes useful examples for things like image transformations (affine), image overlapping, model inference...

# Installation
```
git clone https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples
cd Mediapipe-Halloween-Examples
pip install -r requirements.txt
```
# Original models
Most of the models were taken from Mediapipe: https://google.github.io/mediapipe/solutions/solutions.html. However, the hair segementation model is not currently suppoeted in Python, instead the model from [Kazuhito00's](https://github.com/Kazuhito00) repository was used: https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP

# Examples

 * **Skeleton Pose**:

  ![Mediapipe Skeleton Pose](https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples/blob/main/doc/img/skeleton.gif)
 ```
 python webcamSkeletonPose.py
 ```

 * **Halloween background**:

  ![Mediapipe Halloween background](https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples/blob/main/doc/img/halloween_background.gif)
 ```
 python webcamHalloweenBackground.py
 ```

 * **Exorcist Face Mesh**:

  ![Mediapipe Exorcist Face Mesh](https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples/blob/main/doc/img/exorcist.gif)
 ```
 python webcamFaceMeshExorcist.py
 ```

 * **Pumpkin face**:

  ![Mediapipe Pumpkin face](https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples/blob/main/doc/img/pumpkin_face.gif)
 ```
 python webcamPumpkinFace.py
 ```

 * **Fire Hair**:

  ![Mediapipe Fire Hair](https://github.com/ibaiGorordo/Mediapipe-Halloween-Examples/blob/main/doc/img/fire_hair.gif)
 ```
 python webcamFireHair.py
 ```
 



# References:
Check the header in each of the scripts for the references for each model.
