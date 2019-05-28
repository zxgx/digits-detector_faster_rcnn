# Digits Detector based on Faster RCNN
## **Unfishied yet**.

faster_rcnn.ipynb is a pytorch 1.1 implementation of [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch). I only give a sanity check in this notebook, since it is an object detector trained on VOC dataset, and I implemented it aiming for a better understanding of Faster RCNN.

faster_rcnn_dev.ipynb is my digits detector, which is trained on [SVHN](http://ufldl.stanford.edu/housenumbers/). I haven't finished it yet, because I want to make an optimization for training. Sanity check is given in this notebook to verify the correctness of the model.

# TODO
Many Faster RCNN implementations including the original model only support single-sample training, i.e., the batch size is always 1. This is because the amount of bounding boxes and the spatial size of every sample are not fixed. So I want to implement a Faster RCNN supporting mini-batch training.

# Reference
## detection based on region proposal
Region Proposals: Uijlings et al, “Selective Search for Object Recognition”, IJCV 2013  
Initialization for selective search: P. F. Felzenszwalb and D. P. Huttenlocher. Efficient Graph-Based Image Segmentation. IJCV 2004

R-CNN: Girshick et al, “Rich feature hierarchies for accurate object detection and semantic segmentation”, CVPR 2014  
Fast R-CNN: Girshick, “Fast R-CNN”, ICCV 2015  
Faster R-CNN: Ren et al, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, NIPS 2015
	
## Detection without Proposal
YOLO: Redmon et al, “You Only Look Once: Unified, Real-Time Object Detection”, CVPR 2016
SSD: Liu et al, “SSD: Single-Shot MultiBox Detector”, ECCV 2016