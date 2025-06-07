# MBNet
Hyperspectral image super-resolution based on Mamba and bidirectional
feature fusion network

This repository is implementation of the ["Hyperspectral image super-resolution based on Mamba and bidirectional
feature fusion network"](MBNet)by PyTorch.

Dataset
------
[CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ 
"CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html 
"Harvard"), [Chikusei](https://naotoyokoya.Com/Download.html), are employed to verify the effectiveness of the  proposed SRDNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, The code about data pre-processing in SRDNet (https://github.com/LTTdouble/SRDNet) folder [data pre-processing] or ( https://github.com/qianngli/MCNet/tree/master/data_pre-processing "data pre-processing"). The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
**python 3.9, Pytorch=2.0.1, cuda 12.1, RTX 3090 GPU**

**Please refer to the requirements.txt file for details.**

    pip install requirements.txt
    

After setting the options and dataset paths, you can directly run the training or testing process.

# python train.py

# python test.py

