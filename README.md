## Getting started

### Installation

1. Clone this repository, and we'll call the directory that you cloned as ${POSE_ROOT}

   ```
   git clone https://github.com/zhug777/FaceTR.git
   ```

2. Install PyTorch>=1.8.1 and torchvision>=0.9.1  from the PyTorch [official website](https://pytorch.org/get-started/locally/)

3. Install package dependencies. Make sure the python environment >=3.7

   ```
   pip install -r requirements.txt
   ```
   
4. Make pretrained_models and checkpoints directories under ${POSE_ROOT}:

   ```
   mkdir pretrained_models checkpoints
   ```

5. Download pretrained models for training: [resnet101]('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'), rename it to **resnet101.pth**; [Vision Transformer]('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth'), rename it to **B_16.pth**. And then make them look like this:

   ```
   ${POSE_ROOT}
    `-- pretrained_models
        |-- resnet101.pth
        |-- B_16.pth  
   ```

### Data Preparation

Prepare the LaPa train/val/test dataset and the annotations from [LaPa](https://github.com/lucia123/lapa-dataset). Please download or link them to ${POSE_ROOT}/data/LaPa/, and make them look like this:

```
${POSE_ROOT}/data/LaPa/
|-- test
|   |-- images
|	|   |-- 2569520_1.jpg
|   |   |-- ... 
|   |-- labels
|	|   |-- 2569520_1.png
|   |   |-- ... 
|   |-- landmarks
|	|   |-- 2569520_1.txt
|   |   |-- ... 
|-- train
|   |-- images
|	|   |-- 626176_0.jpg
|   |   |-- ... 
|   |-- labels
|	|   |-- 626176_0.png
|   |   |-- ... 
|   |-- landmarks
|	|   |-- 626176_0.txt
|   |   |-- ... 
`-- val
    |-- images
 	|   |-- 957714_4.jpg
    |   |-- ... 
    |-- labels
 	|   |-- 957714_4.png
    |   |-- ... 
    |-- landmarks
	    |-- 957714_4.txt
        |-- ... 
```

### Traing & Testing

#### Training SETR

```
python train.py --cfg experiments/SETR/SETR_raw.yaml
```

#### Training baseline

```
python train.py --cfg experiments/ResNet/ResNet.yaml
```

#### Training ViT

```
python train.py --cfg experiments/ViT/ViT.yaml
```

### Acknowledgements

Great thanks for these open-source codes：

- [ViT] [https://https://github.com/google-research/vision_transformer](https://https://github.com/google-research/vision_transformer)
- [Pytorch_ViT] [https://https://github.com/lukemelas/PyTorch-Pretrained-ViT](https://https://github.com/lukemelas/PyTorch-Pretrained-ViT)
