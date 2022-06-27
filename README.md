# Segmentation from localization
The implementation of [**Segmentation from localization: a weakly supervised semantic segmentation method for resegmenting CAM**].


## Abstract
Existing studies in weakly supervised semantic segmentation(WSSS) using image-level weak supervision have the following problems: the segmentation results are sparse and target boundaries are inaccurate. To overcome these problems, we propose a locate-then-segment framework, which divides semantic segmentation into class-specific localization process and class-agnosticc segmentation process. We use end-to-end convolutional neural network(CNN) to model the segmentation process, let it learn class-agnostic target boundary features from external pixel-level labels, and use this model to process the weakly supervised semantic segmentation results. Our method can find more complete areas of targets, capture more precise object boundaries, and improve the quality of pseudo-masks. The experimental results show that our method makes up the gap between methods using only image-level supervision and methods using external pixel-level supervision.  Our method achieves 68.8$\%$ and 67.9$\%$ mIoU on val set and test set of Pascal VOC 2012, reaching the current advanced level.  


## Requirements
- Python 3.6
- pytorch 0.4.1
- torchvision 0.2.1
- CUDA 9.0
- GPUs (12GB)


### step
Training and evaluation scripts are in the file.

1. training


```python
!python train_edge.py \ 
    --cocoRoot ${coco path} \ # dir of coco dateset
    --dataType train2017 \
    \
    --batch_size 2 \
    --edge_batch_size 4 \
    --max_epoches 6 \
    --lr 0.001 \
    --edge_lr 1e-4 \
    --num_workers 2 \
    --wt_dec 5e-4 \
    --crop_size 448 \
    \
    --categories ./coco_categories \
    --weights ${weights of cam model} \ # cam's weights
    --edge_weights ${weights of seg model} \ # weight of object segmenter
    \
    --session_name ${session name} \ # session name
    --tblog_dir ./train_in_coco \ # subdir to save logs
    --ret_root ./train_ret \ # subdir to save weights
    --out_root  ${dir to save weights and logs} # dir to save weights and logs
```

2. inference and evaluation
```python
!python infer_and_envalue.py \ # command
    --weights ./SEAM_model/resnet38_SEAM.pth \ # cam's weights
    --edge_weights /home/wujiali/DeepLearning/train_out/edge_out/toEdge20220315_3/train_ret/toEdge20220315_3_edge_055000.pth \ # weight of object segmenter
    --voc12_root /home/wujiali/DeepLearning/dataset/VOC2012 \ # voc_12 dir
    --gt_dir /home/wujiali/DeepLearning/dataset/VOC2012/SegmentationClassAug \ #label dir
    --infer_list voc12/train.txt \ # infer list
    --logfile ./train_list_evallog.txt \ # log file
    --mUIfile ./muilog.txt # log file
```




### Pseudo labels retrain
for the segmentation network. 
- Please see [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) for the implementation in PyTorch.


