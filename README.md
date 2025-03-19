# 3CNet
This project provides the code and results for '3CNet: Cross-modal cooperative correction network for RGB-T semantic segmentation'. 

Thank you for your interest.

## Usage
### Requirements
1. Python 3.8+
2. PyTorch 1.7.0+
3. CUDA 11.7+

and
    pip install -r requirements.txt

>Note:For a detailed environment configuration tutorial you can refer to our CSDN [blog](https://blog.csdn.net/qq_41973051/article/details/128844400?spm=1001.2014.3001.5501)!

### Datasets
<dataset>
|-- <mfnet>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <TH>
        |-- <name1>.<ModalXFormat>
        |-- <name2>.<ModalXFormat>
        ...
    |-- <Labels>
        |-- <name1>.<LabelFormat>
        |-- <name2>.<LabelFormat>
        ...
    |-- train.txt
    |-- test.txt
|-- <PST900>


### Pretrain weights:
Download the pretrained segformer(MiT-B2) [here]() pretrained segformer.

### Config
The parameters of the dataset and the network can be modified through this config file.

## Results
### Results on MFNet (9-class):
![b8958a6ef4b3a76347399d96551a6fb6](https://github.com/user-attachments/assets/e5a970e5-cacf-46f9-8737-c21728f9120e)

| Backbone  | Modal | mIoU   |
|--------|------|--------|
| MiT-B2   | RGB-T   | 60.0   |

The trained model can be downloaded [here](). 

The visualisation of the model predictions can be downloaded [here]()

All code will be made public after the paper is published.
