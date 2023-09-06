# CMPE 249 IntelligentAutonomous Systems


## Syllabus

week1 : Overview, Sensor Systems for autonomous systems

week2 : 


## Project Topic

1. **Algorithm track:** existing challenges and datasets (2D/3D object detection/segmentation, tracking, motion prediction, depth estimation), performance comparison of multiple state-of-art solutions and models for the following datasets:

  * KITTI Vision Benchmark Suite (Basic): https://www.cvlibs.net/datasets/kitti/
  * Waymo open dataset: https://waymo.com/open/
  * Argo open dataset: https://www.argoverse.org/
  * NuScenes: https://www.nuscenes.org/


**Task1: Object Detection**

Dataset : https://www.tensorflow.org/datasets/catalog/waymo_open_dataset

Algorithm : YOLOv8, GroundingDINO, Detectron2, DETR, Faster R-CNN, ByteTrack, RTMDet, EfficientDet,  MobileNet SSD

Reference :

https://roboflow.com/models/object-detection

**Task2: Tracking**

Dataset :

Algorithm : 

1. Open CV object trackers include the BOOSTING, MIL, KCF, CSRT, MedianFlow, TLD, MOSSE, and [GOTURN](http://davheld.github.io/GOTURN/GOTURN.pdf) trackers.
2. DeepSort
3. The Computer Vision Toolbox in MATLAB provides video tracking algorithms, such as continuously adaptive mean shift (CAMShift) and Kanade-Lucas-Tomasi (KLT) for tracking a single object or for use as building blocks in a more complex tracking system.
4. MDNet is a fast and accurate, CNN-based visual tracking algorithm inspired by the R-CNN object detection network. It functions by sampling candidate regions and passing them through a CNN.

Reference

[How to Implement Object Tracking for Computer Vision](https://blog.roboflow.com/object-tracking-how-to/)

[Object Tracking in Computer Vision (2023 Guide)](https://viso.ai/deep-learning/object-tracking/)

[yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort?ref=blog.roboflow.com)

[Object Tracking from scratch with OpenCV and Python](https://youtu.be/GgGro5IV-cs?si=LTXbf9YPknU_r8Y3) + [pysource](https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/)

[What is Object Tracking in Computer Vision?](https://blog.roboflow.com/what-is-object-tracking-computer-vision/)


**Task3: Motion Prediction**

Dataset :

Algorithm : 

**Task4: Depth Estimation**

Dataset : https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

Algorithm : VGG16, Resnet 50, 


2. **System track:** system demonstration of one major component related to autonomous and robotic systems, e.g., deployment and application of existing deep learning models in ROS2 with simulation data or recorded rosbag, evaluate the real-time performance




