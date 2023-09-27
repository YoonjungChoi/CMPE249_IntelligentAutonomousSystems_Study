# HW1: Object Detection 

## 1. Waymo Dataset 

selecte partial dataset from repository from /data/cmpe249-fa22/waymotrain200cocoyolo

```
images_train.txt  has number of classes and objects {0: 9100, 1: 2406, 3: 91}

images_val.txt  has number of classes and objects {0: 1061, 1: 340, 3: 14}

images_test.txt  has number of classes and objects {0: 827, 1: 336, 3: 6}

categories = [{"id": 0, "name": "vehicle"},{"id": 1, "name": "pedestrian"},{"id": 2, "name": "sign"},{"id": 3, "name": "cyclist"}]

Partial dataset does not have sign objects.
```

## 2. Yolov7

### 2.1 YoLo Format Dataset

The same name of png and txt fie are located in this structure folders:

```
dataset/
  train/
    images/
      00001.png
    labels/
      00001.txt
   valid/
   test/
```
### 2.2 Installation & Download

```
pip install torch torchvision torchaudio tensorboard tqdm
pip install opencv-python matplotlib seaborn scipy
pip install --upgrade pyyaml==5.3.1
git clone https://github.com/WongKinYiu/yolov7.git
Download pre-trained model:
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

Created two files:
yolov7/data/kitti_data.yaml
yolov7/cfg/training/yolo7x-kitti.yaml
```

### 2.3 Train/Evaluation/Inference

I did fine-tuning with the yolov7x model. Training log saved in log.train.log file.

Total epochs are 100. Batch size is 8. Image size is (640,640).

```
$ sbatch run.sh 
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     99/99     9.67G   0.02612   0.01135 0.0002898   0.03775        18       640: 100%|██████████| 126/126 [01:35<00:00,  1.32it/s]

[LOG] 100 epochs completed in 2.818 hours.
```


## 3.  Detectron2

### 3.1 COCO format dataset

YoLo format annotation needs to convert COCO format as one JSON file.

### 3.2 Installation & Download

```
git clone 'https://github.com/facebookresearch/detectron2'
python -m pip install -e detectron2
or
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
Download pre-trained model:
!wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
```

### 3.3 Train/Evaluation/Inference

```
#for detection
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
#we take advantages weights and biases from pre-trained model
cfg.MODEL.WEIGHTS = 'model_final_68b088.pkl'
cfg.DATASETS.TRAIN = ("coco_waymo_train")
cfg.DATASETS.TEST = ("coco_waymo_valid")
#GPU 1 with 12GB
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
#We have 1000 train samples: 2batch*500 iters = 1 epoch.
#So for  50 epoch, MAX_TER = 500*50 = 25000
cfg.SOLVER.MAX_ITER = 25000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
#Every 2000 iters, checkpoint will be saved.
#cfg.SOLVER.CHECKPOINT_PERIOD = 2000
I trained the model up to 16,000 iterations, so 32 epochs.
```


## Reference.

[1] Detectron2 Github https://github.com/facebookresearch/detectron2 
[2] Detectron2 Tutorial.ipynb Reference Colab Notebook 
[3] Detectron2 training with custom set Youtube
[4] Detectron2 https://blog.roboflow.com/how-to-train-detectron2/
[5] Yolov7 Github https://github.com/WongKinYiu/yolov7
[6] Convert data format https://www.kaggle.com/code/siddharthkumarsah/ convert-yolo-annotations-to-coco-pascal-voc
[7]  detectron2 explaination ttps://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd
[8] KaiKai Github https://github.com/lkk688/DeepDataMiningLearning










