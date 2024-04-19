# uav-pose-estimaion
For UAV tracking, detecting the UAV and its center point alone is not sufficient. After the UAV is detected, the center, right wing and left wing can be determined with the pose estimation method, and UAV tracking can be made more possible with the tilt information of the UAV. This code only contains a demo for the visual tracking part.

![uav_pose](https://github.com/KARAASLAN-AI/uav-tracking-with-pose-estimaion/blob/main/images/Untitled%20(2).gif)

##  Yolov4 and Deeplabcut Pose Tracking

In this application, UAV was detected using yolov4. Pose estimation was done using deeplabcut. With the opencv-GPU installation, drone detection and pose estimation were carried out at over 60 fps.

> For Opencv-CUDA installation follow this link: <br/>
https://www.youtube.com/watch?v=YsmhKar8oOc  <br/>
> For deeplabcut : <br/> https://github.com/DeepLabCut/DeepLabCut

```
python yolov4-detect_pose-track.py
```
