# CMPE 249 IntelligentAutonomous Systems

[HomeWork 1: Object Detection based on YoloV7 and detectron2 algorithms] 

[Project Details Task4: Depth Estimation](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/tree/main/task4_depth_estimation)



## HPC Tunneling With local Jupyter 

The GPU node does not have internet access.
If you wish to access the Jupyter web interface in your local browser, you can set up a tunnel from your local computer to the HPC headnode
and then create another tunnel from the HPC headnode to the GPU node.

```
conda activate env
conda install -c conda-forge jupyterlab

ssh -L 10001:localhost:10001 0107xxx@coe-hpc1.sjsu.edu #from your local computer to HPC headnode, you can use any port number (10001)
ssh -L 10001:localhost:10001 0107xxx@g7 #in HPC head node to gpu node
activate python virtual environment, e.g., conda activate xxx
jupyter lab --no-browser --port=10001 #start the jupyter lab on port 10001 (the port should be the same port used for tunnel)

```

## Project Topics

### Algorithm track
existing challenges and datasets (2D/3D object detection/segmentation, tracking, motion prediction, depth estimation), performance comparison of multiple state-of-art solutions and models for the following datasets:

  * KITTI Vision Benchmark Suite (Basic): https://www.cvlibs.net/datasets/kitti/
  * Waymo open dataset: https://waymo.com/open/
  * Argo open dataset: https://www.argoverse.org/
  * NuScenes: https://www.nuscenes.org/

**Task1: Object Detection**

 Dataset : [waymo-dataset](https://www.tensorflow.org/datasets/catalog/waymo_open_dataset)

 Algorithm : **YOLOv8**, GroundingDINO, **Detectron2**, DETR, Faster R-CNN, ByteTrack, RTMDet, EfficientDet,  MobileNet SSD

 Reference : [Description](https://roboflow.com/models/object-detection)

**Task2: Object Tracking**

1. Open CV object trackers include the BOOSTING, MIL, KCF, CSRT, MedianFlow, TLD, MOSSE, and [GOTURN](http://davheld.github.io/GOTURN/GOTURN.pdf) trackers.
2. DeepSort
3. The Computer Vision Toolbox in MATLAB provides video tracking algorithms, such as continuously adaptive mean shift (CAMShift) and Kanade-Lucas-Tomasi (KLT) for tracking a single object or for use as building blocks in a more complex tracking system.
4. MDNet is a fast and accurate, CNN-based visual tracking algorithm inspired by the R-CNN object detection network. It functions by sampling candidate regions and passing them through a CNN.

Reference:  [How to Implement Object Tracking for Computer Vision](https://blog.roboflow.com/object-tracking-how-to/) , [Object Tracking in Computer Vision (2023 Guide)](https://viso.ai/deep-learning/object-tracking/) , [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort?ref=blog.roboflow.com), [Object Tracking from scratch with OpenCV and Python](https://youtu.be/GgGro5IV-cs?si=LTXbf9YPknU_r8Y3) + [pysource](https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/) , [What is Object Tracking in Computer Vision?](https://blog.roboflow.com/what-is-object-tracking-computer-vision/)


**Task3: Motion Prediction**



**Task4: Depth Estimation**

Dataset : [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 

Algorithm : [VA-DepthNet](https://github.com/cnexah/VA-DepthNet/tree/main) , [DiffusionDepth](https://github.com/duanyiqun/DiffusionDepth) , [BTS](https://github.com/cogaplex-bts/bts)

Reference : [2023.SEP] 3rd place [VAdepthNet](https://github.com/cnexah/VA-DepthNet), 13rd place [NeWCRFs](https://github.com/aliyun/NeWCRFs/tree/master), 27rd place [BTS](https://github.com/cleinc/bts)

[Lidar Visualization](https://github.com/enginBozkurt/Visualizing-lidar-data/blob/master/Kitti-Dataset.ipynb) , [HuggingFace Docs](https://huggingface.co/docs/datasets/main/en/depth_estimation)

### System track 

system demonstration of one major component related to autonomous and robotic systems, e.g., deployment and application of existing deep learning models in ROS2 with simulation data or recorded rosbag, evaluate the real-time performance











