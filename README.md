# Combining Invariant Information Learning with K-means Clustering for Self-supervised Image Classification

This repository contains PyTorch code for the the final project of Computer Vison for Jiahui Li (jl10005@nyu.edu) and Zimo Li (zl2521@nyu.edu). It's forked based on repository for [IIC paper](https://arxiv.org/abs/1807.06653).

`train.sh` is the script we use to train self-supervised model on prince.

and `fine_tune.sh` is the script we use to train fine-tune model on prince.

# Package dependencies
Listed <a href="https://github.com/xu-ji/IIC/blob/master/package_versions.txt">here</a>. You may want to use e.g. virtualenv to isolate the environment. It's an easy way to install package versions specific to the repository that won't affect the rest of the system.

# Running on your own dataset
You can either plug our loss (paper fig. 4, <a href="https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py#L6">here</a> and <a href="https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/IID_losses.py#L86">here</a>) into your own code, or change scripts in this codebase. Auxiliary overclustering makes a large difference (paper table 2) and is easy to implement, so it's strongly recommend even if you are using your own code; the others settings are less important. To edit existing scripts to use different datasets see <a href="https://github.com/xu-ji/IIC/issues/8">here</a>.
