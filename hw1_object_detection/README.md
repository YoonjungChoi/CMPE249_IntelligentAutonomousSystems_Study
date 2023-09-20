# Object Detection


## Installation & Download
```
pip install torch torchvision torchaudio tensorboard tqdm
pip install opencv-python matplotlib seaborn scipy
pip install --upgrade pyyaml==5.3.1
git clone https://github.com/WongKinYiu/yolov7.git
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

git clone 'https://github.com/facebookresearch/detectron2'
python -m pip install -e detectron2
```


## 1. YoLoV7 : https://github.com/WongKinYiu/yolov7

#### Preparing Dataset

```
pip install labelImg
```
Using labelImg, create rectangle objects annotation

![Screen Shot 2023-09-18 at 10 19 46 AM (2)](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/1a5be898-d701-4d84-9357-b6055b9e8f2b)

  
## 2. Detectron2: https://github.com/facebookresearch/detectron2

#### Preparing Dataset

```
pip install labelme
```
Using labelme, create polygon objects annotation

![Screen Shot 2023-09-18 at 2 50 05 PM (2)](https://github.com/YoonjungChoi/CMPE249_IntelligentAutonomousSystems_Study/assets/20979517/54100a88-c508-403c-8108-f6c13fb3dff0)
