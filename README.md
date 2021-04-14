# Learning to Represent Whole Slide Images by Selecting Cell Graphs of Patches
Yinan Zhang, Beril Besbinar, Pascal Frossard

## Overview
This code is developed for the paper "Learning to Represent Whole Slide Images by Selecting Cell Graphs of Patches". In this paper, we propose an algorithm for classifying multi-graphs with automatic graph selection. Our framework introduces stochastic layers with discrete random variables into traditional graph neural networks and is end-to-end differentiable.

![](img/pipeline.png)

## Requirements
ipywidgets                7.5.1  
matplotlib                3.1.3  
networkx                  2.4  
numpy                     1.18.1  
pandas                    1.0.1  
pytorch                   1.4.0   
scikit-learn              0.22.2  
scipy                     1.4.1  
seaborn                   0.10.0  
tqdm                      4.43.0  
Especially, please install [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) as well.

## Usage

```
# 1. Create training / validation / test dataset using Data-Generation.ipynb using a fixed random seed 123 / 124 / 125 respectively.  

# 2. Pre-trained the graph embedding module and the classification layer using PatientClassifcation-pretrain.ipynb.

# 3. Assess pretraining results using Plot-results-pretrain.ipynb.

# 4. Load pre-trained parameters and jointly learn graph embedding and selection using PatientClassifcation-joint.ipynb. Note that you should choose a directory to save model parameters at different epochs.

# 5. Assess classifcation and selection performance using Plot-results-joint.ipynb.
```

## Result
We  experimented with different aggregators, different architectures and soft and hard selection of patches, and the best performance is obtained with element-wise mean aggregator and soft selection mechanism. The confusion matrix below corresponding to the average of 100 realizations of sampling 2 graphs for 10 patients. The corresponding percentage of selecting both ground truth discriminative patches correctly, selecting one out of two patches correctly and not being able to select any right patches correctly are 49.1%, 50.8% and 0.1%, respectively.

<img src=img/max-soft-mean.png width="50" height="50">

## License
This project is licensed under the MIT License.

