# Depth Estimation

## Project milestone
```
1. A clear list of what you have achieved so far, add your code links. Include the current status of your planned modules. Are they fully functional, partially functional, or still in the development phase?


2. Offer a detailed description of the identified 'baseline' modules. Describe the functionality and purpose of these baseline modules. What role do they play in the project.  If applicable, share information about how these baseline modules were implemented or integrated into your project via an additional video demo or screenshots.


3. References to any sources you have used during the project.


4. Any challenges you have encountered along the way.

5. Your plans to overcome these challenges and move forward with the project.

```

## 1. setup Environments
```
conda create -n env
conda activate env
conda install torch torchvision cudatoolkit
pip install tqdm timm
```

```
#check version
import torch
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())
```

For HPC
```
#interactive mode
srun -p gpu --gres=gpu --ntasks 1 --nodes 1 --cpus-per-task 4 --pty /bin/bash
conda activate env
```

## 2. Dataset: KITTI 

[Dataset KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

The camera ID means 02 for left, 03 for right camera. The "groundtruth" describes our annotated depth maps image_02 is the depth map for the left camera.

The image IDs start at 5 because we accumulate 11 frames which is +-5 around the current frame

First, we need to download all datasets, and then unzip as their instruction.

```

#Output groundtruth dataset
Annotated depth mas data set
    train/2011_10_03_drive_0042_sync/proj_depth/groundtruth/image_02/0000000005.png
    val/2011_10_03_drive_0042_sync/proj_depth/groundtruth/image_02/
Raw LiDaR scans data set
    train/2011_10_03_drive_0042_sync/proj_depth/velodyne_raw/image_02/
    val/2011_10_03_drive_0042_sync/proj_depth/velodyne_raw/image_03/
#Input dataset
Raw dataset download script (RGB Image)
	  2011_10_03/2011_10_03_drive_0042_sync/image_02/

#For challenge
Selected validation and test data sets
  test_depth_prediction_anonymous/image/
  test_depth_prediction_anonymous/intrinsics/
  val_selection_cropped/groundtruth_depth/
  val_selection_cropped/image/
  val_selection_cropped/intrinsics/
  val_selection_cropped/velodyne_raw/
```

Figure 1.  2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000095.png

![0000000095](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/54e75cd2-59c7-4168-9b15-3e0803d3895f)

Figure 2. 2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000095.png

![0000000095](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/447aeb25-670b-4050-934f-1e16d8f4ccd1)



**KITTI Eigen Split**  - [link](https://stackoverflow.com/questions/63512296/kitti-eigen-split)

Many papers use Kitti Eigen Split, which means we have to combine train and val in the ground truth files.
```
-- kitti_raw/2011_10_03/2011_10_03_drive_0042_sync/image_02/
-- gt/2011_10_03_drive_0042_sync/proj_depth/velodyne_raw/image_02/
-- gt/2011_10_03_drive_0042_sync/proj_depth/groundtruth/image_02/
```

They has already splited and saved as data_splits/files..

**Exploring Dataset**

Reference - [KITTI-Dataset](https://github.com/alexstaravoitau/KITTI-Dataset/blob/master/kitti-dataset.ipynb)


## 3. Algorithm

### 3.1 VA-DepthNet: A Variational Approach to Single Image Depth Prediction

Eval on challenge validation dataset

```
(newDepth) [013907062@g5 VA-DepthNet]$ python vadepthnet/eval.py configs/yoon_arguments_eval_kittieigen.txt
/home/013907062/.conda/envs/newDepth/lib/python3.11/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3483.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
== Total number of parameters: 263110761
== Total number of learning parameters: 263110761
== Model Initialized
== Loading checkpoint 'ckpts/vadepthnet_eigen.pth'
== Loaded checkpoint 'ckpts/vadepthnet_eigen.pth'
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [07:27<00:00,  2.24it/s]
Computing errors for 1000 eval samples , post_process:  False
  silog, abs_rel,   log10,     rms,  sq_rel, log_rms,      d1,      d2,      d3
 6.5207,  0.0461,  0.0198,  1.9626,  0.1426,  0.0714,  0.9802,  0.9967,  0.9991
```


- data_splits/kitti_official_valid_with_gt.txt

- VA-DepthNet/test.py

```
(base) [013907062@coe-hpc1 VA-DepthNet]$ tail log.eval.log
Missing gt for ../dataset/raw/./2011_09_30/2011_09_30_drive_0016_sync/image_02/data/0000000275.png
Missing gt for ../dataset/raw/./2011_09_30/2011_09_30_drive_0016_sync/image_02/data/0000000000.png
Missing gt for ../dataset/raw/./2011_09_30/2011_09_30_drive_0018_sync/image_02/data/0000000000.png
Missing gt for ../dataset/raw/./2011_09_30/2011_09_30_drive_0027_sync/image_02/data/0000000000.png
Missing gt for ../dataset/raw/./2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000000.png
Missing gt for ../dataset/raw/./2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000000.png
100%|██████████| 697/697 [04:43<00:00,  2.46it/s]
Computing errors for 652 eval samples , post_process:  False
  silog, abs_rel,   log10,     rms,  sq_rel, log_rms,      d1,      d2,      d3
 6.8172,  0.0502,  0.0219,  2.0927,  0.1479,  0.0758,  0.9773,  0.9974,  0.9994
```

Results:

Input:
![val_selection_cropped_2011_10_03_drive_0047_sync_image_0000000791_image_03](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/75b76b43-742c-48b2-aad5-2b9940e2026b)

Prediction:

![val_selection_cropped_2011_10_03_drive_0047_sync_image_0000000791_image_03](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/cf056d1e-377e-4300-b891-14500f424ecc)

![val_selection_cropped_2011_10_03_drive_0047_sync_image_0000000791_image_03](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/d52cc0a1-952e-4c4e-9815-203810f3e883)

Ground Truth:

![2011_10_03_drive_0047_sync_groundtruth_depth_0000000791_image_03](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/0512aa55-dbf1-467d-8346-8bf2dab39e52)





Evaluation Metrics: Scale Invariant Logarithmic error, Relative Squared error, Relative Absolute Error, Root Mean Squared error




## Appendix


Backbone - pretrained Swin Transformer with ImageNet

**What is a Vision Transformer (ViT) ?**

Image classification is a fundamental task in computer vision. Over the years, CNNs like YOLOv7 have been the state-of-the-art method for image classification. However, recent advancements in transformer architecture, which was originally introduced for natural language processing (NLP), have shown great promise in achieving competitive results in image classification tasks.

The self-attention layer calculates attention weights for each pixel in the image based on its relationship with all other pixels, while the feed-forward layer applies a non-linear transformation to the output of the self-attention layer. The multi-head attention extends this mechanism by allowing the model to attend to different parts of the input sequence simultaneously. [Reference]

It splits an image into fixed-sized patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, it uses the standard approach of adding extra learnable ‘classification token’ to the sequence. [Reference](https://viso.ai/deep-learning/vision-transformer-vit/)

**What is a Swin Transformer?**

Swin Transformer constructs a hierarchical representation by starting from small-sized patches and gradually merging neighboring patches in deeper Transformer layers.  [Reference](https://medium.com/dair-ai/papers-explained-26-swin-transformer-39cf88b00e3e)







