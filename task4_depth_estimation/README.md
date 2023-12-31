# Depth Estimation


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
srun -p gpu --gres=gpu --pty /bin/bash

#tunneling
ssh -L 10007:localhost:10007 SJSU_ID@g8

#activate env
conda activate env

#open jupyter
jupyter lab --no-browser --port=10007

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


## 3. VA-DepthNet Algorithm

### 3.1 Architecture 

We evaluated baseline models with the KITTI Eigen TEST set, and the results show the below.

```
pretrained model by authors with Swin Transformer Large backbone
 silog, 	abs_rel,       log10,     rms, 	 sq_rel,   log_rms,      d1,           d2,      	d3
 6.5207,  0.0461,  0.0198,  1.9626,  0.1426,  0.0714,  0.9802,  0.9967,  0.9991

our reproduced model with Swin Transformer Large backbone. Best silog evaluation at 98000 steps  of 16 epoch.
  silog, 	abs_rel,       log10,     rms, 	 sq_rel,   log_rms,      d1,           d2,      	d3
 6.8139,  0.0515,  0.0225,  2.0971,  0.1497,  0.0766,  0.9774,  0.9973,  0.9994

our reproduced model with Swin Transformer Tiny backbone. Best silog evaluation at 23400 steps of 40/50 epochs
 silog, 	abs_rel,       log10,     rms, 	 sq_rel,   log_rms,      d1,           d2,      	d3
 7.7805,  0.0574,  0.0249,  2.3042,  0.1873,  0.0860,  0.9673,  0.9959,  0.9991
```

We faced Memory Issue, so we decided to use Tiny Backbone, and we need to modify some params to use Tiny backbone.

From Large Backbone,  4 feature maps  X5: [ BatchSize, **1536**, 11, 38 ],  X4: [BatchSize, **768**, 22, 76 ], X3: [ BatchSize, **384**, 44, 152 ]
 X2: [ BatchSize, **192**, 88, 304 ] are extracted.

From Tiny Backbone, 4 feature maps X5: [ BatchSize, **768**, 11, 38 ], X4: [ BatchSize, **384**, 22, 76 ],  X3: [ BatchSize, **192**, 44, 152 ]
X2: [ BatchSize, **96**, 88, 304 ] are extracted.

Here is a diagram we draw to understand architecuture.

<img width="945" alt="architecture_overview" src="https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/147fa045-428a-488a-b14c-d5463e83e105">


### 3.2 Exploring eval.py test.py pre-trained model

Eval on challenge validation dataset, again..

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

### 3.3 Experiments

#### 3.3.1 Experiment 1 & 2 redesign of Refine module with Res2Net module.

Previous refine module has a simple structure, so we wanted to redesign the refine module with the Res2Net module. The Res2Net module has multiscale ability. This multi-feature aggregation allows the integration of multi-scale features at each layer, and help network capture both local and global contextual information.

This diagram shows original Refine module of VAdepthNet

![CMPE 249 VAdepthNet-Refine drawio](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/3ac36b1e-623a-4c84-a8e0-3542ee9298f6)

This diagram shows Res2Net + Refine modules variant that we proposed.

<img width="953" alt="experiment1_2_refine_res2net" src="https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/5f953bf9-8ef4-4f08-8fed-eb6021646fb1">

#### 3.3.2 Experiment 3 redesign of metric module with SE module.

Original SE module has the same number of channels before/after squeeze and excitation. However, this architecture’s metric module has a single channel. We doubt its effectiveness, because the SE module's output channel has different feature information. Thus, we redesigned the metric module a slight differently .

This diagram shows original metric module of VAdepthNet

<img width="502" alt="metricmodule" src="https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/09d2d50d-5e8c-4eff-bca0-074bd09edeb7">

This diagram shows SE + Metric modules that we proposed.

<img width="467" alt="metric_re" src="https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/94924bce-7b37-4cda-847a-5425a813068d">


#### 3.3.3 Experiment 4 Freeze layer of backbone.

So far, we updated all parameters, but ‘freeze’ technique is obvious to cut down computational time while losing not much accuracy. Thus, we experimented freeze patch embedding, absolute position embedding, first two layers (SwinTransformerTiny has 4 layers).

```
== SwinTransformer Total number of parameters : 30,088,890
== SwinTransformer Total number of learning parameters: 27,216,612
== Total number of parameters: 70,983,567
== Total number of learning parameters: 68,111,289
```

#### 3.3.4 Experiment 5 Change CNN-based backbone.

We can change CNN-based backbone, which has more parameters.

```
Swin Transformer Tiny
== Total number of parameters: 70983567
== Total number of learning parameters: 70,983,567
X5: [ BatchSize, 768, 11, 38 ]
X4: [ BatchSize, 384, 22, 76 ]
X3: [ BatchSize, 192, 44, 152 ]
X2: [ BatchSize, 96, 88, 304 ]


ResNet50 (resnet50-0676ba61.pth)
== Total number of parameters: 106,512,669
== Total number of learning parameters: 106,512,669
X5: [ BatchSize, 2048, 11, 38 ]
X4: [ BatchSize, 1024, 22, 76 ]
X3: [ BatchSize, 512, 44, 152 ]
X2: [ BatchSize, 256, 88, 304 ]
```

### 3.4 Results

Every experiment has better performance than baseline except experiment 5 using CNN-based backbone Resnet50. Experiment 1 modifying the Refine module with Res2Net without residual connection has better performance than Experiment 2 with residual connection. Also, Experiment 3 does not have much improvement compared to Baseline, not better and not worse. Experiment 4 freezing layers have the best performance among those experiments. Since freezing former layers still has generic features trained by ImageNet Dataset and KITTI dataset is the specific task focused dataset, its valuable features it has learned would make the network robust in the new task.

<img width="1022" alt="results" src="https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/c8dd2bc5-3360-4d25-beee-4006d16c2b0b">




## Appendix


Backbone - pretrained Swin Transformer with ImageNet

**What is a Vision Transformer (ViT) ?**

Image classification is a fundamental task in computer vision. Over the years, CNNs like YOLOv7 have been the state-of-the-art method for image classification. However, recent advancements in transformer architecture, which was originally introduced for natural language processing (NLP), have shown great promise in achieving competitive results in image classification tasks.

The self-attention layer calculates attention weights for each pixel in the image based on its relationship with all other pixels, while the feed-forward layer applies a non-linear transformation to the output of the self-attention layer. The multi-head attention extends this mechanism by allowing the model to attend to different parts of the input sequence simultaneously. [Reference]

It splits an image into fixed-sized patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, it uses the standard approach of adding extra learnable ‘classification token’ to the sequence. [Reference](https://viso.ai/deep-learning/vision-transformer-vit/)

**What is a Swin Transformer?**

Swin Transformer constructs a hierarchical representation by starting from small-sized patches and gradually merging neighboring patches in deeper Transformer layers.  [Reference](https://medium.com/dair-ai/papers-explained-26-swin-transformer-39cf88b00e3e)







