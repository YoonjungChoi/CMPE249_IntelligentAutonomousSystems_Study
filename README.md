# CMPE 249 IntelligentAutonomous Systems

[HomeWork 1: Object Detection based on YoloV7 and detectron2 algorithms] 

[HomeWork 2: A star Algorithm]

[Project Details Task4: Depth Estimation](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/tree/main/task4_depth_estimation)

**Project - Depth Estimation**

Dataset : [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 

Algorithm : [VA-DepthNet](https://github.com/cnexah/VA-DepthNet/tree/main) , [DiffusionDepth](https://github.com/duanyiqun/DiffusionDepth) , [BTS](https://github.com/cogaplex-bts/bts)

Reference : [2023.SEP] 3rd place [VAdepthNet](https://github.com/cnexah/VA-DepthNet), 13rd place [NeWCRFs](https://github.com/aliyun/NeWCRFs/tree/master), 27rd place [BTS](https://github.com/cleinc/bts)

[Lidar Visualization](https://github.com/enginBozkurt/Visualizing-lidar-data/blob/master/Kitti-Dataset.ipynb) , [HuggingFace Docs](https://huggingface.co/docs/datasets/main/en/depth_estimation)

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

## Study on Computer Vision


