Hierarchical X-Ray Detection Model (Swin+PVT) Repository
Introduction
This repository contains a hierarchical and lightweight Transformer-based model for illegal item detection in X-ray images. The project implements the approach described in "Hierarchical and Lightweight Transformer-based Model for Illegal Item Detection in X-Ray Images" (Swin+PVT backbone with pruning, quantization, and distillation on the SIXray dataset). It is organized to demonstrate the paper's structure and training workflow.
 Key features:
Transformer Backbone: Uses a hybrid Swin Transformer + Pyramid Vision Transformer (PVT) backbone for multi-scale feature extraction.
Lightweight Design: Incorporates filter pruning and INT8 quantization to reduce model size and inference cost​

Knowledge Distillation: A small Swin-PVT student model is trained to mimic a ResNet50-FPN teacher detector, improving accuracy without adding complexity​

One-Stage Detector: Uses an FPN neck and a RetinaNet-style one-stage head for prohibited item detection​

Benchmark (SIXray): Evaluated on the SIXray dataset (6 classes of threats: Gun, Knife, Wrench, Pliers, Scissors, Hammer)​
, with results indicating on-par mAP with the teacher model at a fraction of the model size, and faster inference.

The code is built with PyTorch 1.8.1,mmcv-full ==1.3.17 and  MMDetection== 2.20 (OpenMMLab). 

Important! :Better run in CUDA 11.1 and with Visual studio 2019(2022 cant complie the mmcv and mmdetection) Also the latest version of mmdetection and mmcv have poor support for project swin tran, that means you have to install mmcv-full and mmdetection by humanly complie it) And Apex-master from NVIDIA is needed. Linux and Macos are better than windows(If you run in windows will have so many bugs, and there's problem when you use WINDOWS 10,  WINDOWS 11 is better.




