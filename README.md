# Deterministic-and-Stochastic-Synthesis-Network
Codes and models for the SIGGRAPH Asia 2018 paper "Image Super-Resolution via Deterministic-Stochastic Synthesis and Local Statistical Rectification".

# Deterministic and Stochastic Synthesis

By [Weifeng Ge](https://i.cs.hku.hk/~wfge/), [Yizhou Yu](http://i.cs.hku.hk/~yzyu/)

Department of Computer Science, The University of Hong Kong

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Pipeline](#pipeline)
0. [Codes and Installation](#codes-and-installation)
0. [Models](#models)
0. [Results](#results)

### Introduction

This repository contains the codes and models described in the paper "Image Super-Resolution via Deterministic-Stochastic Synthesis and Local Statistical Rectification"(https://arxiv.org/pdf/1809.06557.pdf). These models are trained and tested on the dataset of [NTIRE 2017](http://www.vision.ee.ethz.ch/ntire17/) DIV2K super resolution track.

**Note**

0. All algorithms are implemented based on the deep learning framework [Caffe](https://github.com/BVLC/caffe).
0. Please add the additional layers used into your own Caffe to run the training codes.

### Citation

If you use these codes and models in your research, please cite:

       @inproceedings{ge2018image,
               title={Image super-resolution via deterministic-stochastic synthesis and local statistical rectification},
               author={Ge, Weifeng and Gong, Bingchen and Yu, Yizhou},
               booktitle={SIGGRAPH Asia 2018 Technical Papers},
               pages={260},
               year={2018},
               organization={ACM}
       }
