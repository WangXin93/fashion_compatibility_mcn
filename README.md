# Outfit Compatibility Prediction and Diagnosis with Multi-Layered Comparison Network

![diagnosis](./exp/diagnosis.png)

## News

* [2019-07-02] Our work has been accepted by ACM Multimedia 2019!

## Introduction

Outfit diagnosis means figuring out the incompatibility in the fashion outfit. While several works explore predicting the compatibility of outfit, none of them show interpretation for their prediction, which is helpful in giving advice to outfits composed by users and automatically revision. In this work, we introduce a framework named multi-layered comparison network (MCN) for both predicting and diagnosing the outfit. The key idea of MCN is learning the compatibility score from all type-specified pairwise similarities between items. We implement the diagnosis by using the backpropagated gradients to approximate the importance of each input to the incompatibility.

## Contents of this repository

* [mcn](./mcn): Main program source code
* [data](./data): **Polyvore-T** datasets based on [Polyvore](https://github.com/xthan/polyvore-dataset).
* [baselines](./baselines): Compared baselines in our experiment
* [exp](./exp): Experiment details, scripts and results etc.
* [ref](./ref): Paper files of the related works

## Requirements

Ubuntu 16.04, NVIDIA GTX 1080Ti (for batch size 16), python >= 3.5.2

```
torch>=0.4.1
torchvision
networkx
```

## Usage

1. Download the original [Polyvore]() dataset, then unzip the file and put the `image` directory into `data` folders (or you can create a soft link for it).

2. Train

   ```sh
   cd mcn
   python train.py
   ```

4. Evaluate

   ```
   python evaluate.py
   ```

5. Diagnose

   ```
   jupyter notebook # then open diagnosis.ipynb
   ```

   

## Prediction Performance

Pretrained model weights can be found in the link.

|                                                              |    AUC    |   FITB    |
| :----------------------------------------------------------- | :-------: | :-------: |
| Pooling                                                      |   88.35   |   57.28   |
| Concatenation                                                |   83.40   |   52.91   |
| Self-attention                                               |   79.65   |   48.60   |
| [BiLSTM](https://drive.google.com/open?id=1WaUP0X-ytZ05HYzeHmdBSzT9gcjF1c46) |   74.82   |   46.02   |
| [CSN](https://drive.google.com/open?id=1EYwtJBRMFxRDzQs7JNYQhp2TpRF2fw9r) |   84.90   |   57.06   |
| [Ours](https://drive.google.com/open?id=1--CfX5LMTxrdxSL_xkDb6MBaNRcAXeXg) | **91.90** | **64.35** |
