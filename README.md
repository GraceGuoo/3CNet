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

<details>
  <summary>dataset</summary>
  <ul>
    <li>mfnet/
        <ul>
        <li>RGB/
      <ul>
        <li>name1.png</li>
        <li>name2.png</li>
      </ul>
    </li>
    <li>TH/
      <ul>
        <li>……</li>
      </ul>
    </li>
    <li>Labels/
      <ul>
        <li>……</li>
      </ul>
    </li>
  </ul>
    </li>
    <li>PST900/
      <ul>
        <li>……</li>
      </ul>
    </li>
  </ul>
</details>


### Pretrain weights:
Download the pretrained segformer(MiT-B2) [here](https://pan.baidu.com/s/1UqfoK30iMcOYKym0dxQPzA?pwd=3cnt) pretrained segformer.

### Config
The parameters of the dataset and the network can be modified through this config file.

## Results
### Results on MFNet (9-class):
![b8958a6ef4b3a76347399d96551a6fb6](https://github.com/user-attachments/assets/e5a970e5-cacf-46f9-8737-c21728f9120e)

| Backbone  | Modal | mIoU   |
|--------|------|--------|
| MiT-B2   | RGB-T   | 60.0   |

The trained model can be downloaded [here](https://pan.baidu.com/s/1ykLytOpqHO8hkuewflBjXw?pwd=3cnt).

The visualisation of the model predictions can be downloaded [here](https://pan.baidu.com/s/11sMy3p5f0M9c41zPXaif1Q?pwd=3cnt)

All code will be made public after the paper is published.
