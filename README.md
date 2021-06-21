# ebms_3dod

![overview image](ebms_3dod.jpg)

Official implementation (PyTorch) of the paper: \
**Accurate 3D Object Detection using Energy-Based Models**, CVPR Workshops 2021 [[arXiv]](https://arxiv.org/abs/2012.04634) [[project]](http://www.fregu856.com/publication/ebms_3dod/). \
[Fredrik K. Gustafsson](http://www.fregu856.com/), [Martin Danelljan](https://martin-danelljan.github.io/), [Thomas B. Schön](http://user.it.uu.se/~thosc112/). \
_We apply energy-based models p(y|x; theta) to the task of 3D bounding box regression, extending the recent energy-based regression approach from 2D to 3D object detection. This is achieved by designing a differentiable pooling operator for 3D bounding boxes y, and adding an extra network branch to the state-of-the-art 3D object detector SA-SSD. We evaluate our proposed detector on the KITTI dataset and consistently outperform the SA-SSD baseline, demonstrating the potential of energy-based models for 3D object detection._

[Youtube video](https://youtu.be/7JP6V818bh0) with qualitative results: \
[![demo video with qualitative results](https://img.youtube.com/vi/7JP6V818bh0/0.jpg)](https://youtu.be/7JP6V818bh0)

If you find this work useful, please consider citing:
```
@inproceedings{gustafsson2020accurate,
  title={Accurate 3D Object Detection using Energy-Based Models},
  author={Gustafsson, Fredrik K and Danelljan, Martin and Sch{\"o}n, Thomas B},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2021}
}
```









## Acknowledgements

- The code is based on [SA-SSD](https://github.com/skyhehe123/SA-SSD) by [@skyhehe123](https://github.com/skyhehe123).









## Index
TODO!
***
***
***










***
***
***

This repository is a work in progress, the code is currently being uploaded (June 20 2021).





































***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***
***


- NOte that I had to use exactly the same versions for pytorch (1.1.0) and spconv (1.0) as in https://github.com/skyhehe123/SA-SSD for the code to work.
*
*
*
*
*
- $ pip install opencv-python
- $ pip install Shapely
- $ pip install mmcv==0.2.14 (NOTE! It did not work with the latest version)
- $ pip install terminaltables
- $ apt-get update
- $ apt-get install -y libsm6 libxext6 libxrender-dev
- $ pip install opencv-python
- $ pip install torch==1.1.0 torchvision==0.3.0 (NOTE! pytorch 1.1.0)
- $ pip install numba
- $ pip install Cython
- $ pip install pycocotools
- $ pip install scikit-image
*
*
- Install spconv 1.0 (NOTE! spconv 1.0):
- - $ cd ebms_3dod/3dod
- - $ git clone https://github.com/traveller59/spconv.git --recursive
- - $ cd spconv
- - $ git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 (this is the commit corresponding to spconv 1.0)
- - $ apt-get update
- - $ apt-get install libboost-all-dev
- - $ python setup.py bdist_wheel
- - $ cd dist
- - $ pip install spconv-1.0-cp36-cp36m-linux_x86_64.whl (spconv-1.0-cp36-cp36m-linux_x86_64.whl was the name of the file at least for me)
*
*
- $ cd ebms_3dod/3dod
- $ pip install pybind11
- $ cd mmdet/ops/points_op
- $ python setup.py build_ext --inplace
- $ cd mmdet/ops/pointnet2
- $ python setup.py build_ext --inplace
- $ cd mmdet/ops/iou3d
- $ python setup.py build_ext --inplace
*
*
- Download the pretrained SA-SSD model from https://drive.google.com/file/d/1WJnJDMOeNKszdZH3P077wKXcoty7XOUb/view, place the file epoch_50.pth in ebms_3dod/3dod.
*
*
- Create the folders ebms_3dod/3dod/data and ebms_3dod/3dod/data/KITTI
- Download the KITTI dataset, place the "ImageSets" and "object" folders in ebms_3dod/3dod/data/KITTI
*
*
- Create cropped point clouds and sample for data augmentation:
- - Create the folder ebms_3dod/3dod/data/KITTI/object/training/velodyne_reduced
- - Create the folder ebms_3dod/3dod/data/KITTI/object/testing/velodyne_reduced
- - $ cd ebms_3dod/3dod
- - $ python create_data.py

























*
*
*
*
*
- Train model on KITTI train:
- - $ cd ebms_3dod/3dod
- - $ python train.py configs/car_cfg20.py

- Evaluate model on KITTI val:
- - $ cd ebms_3dod/3dod
- - $ python eval.py configs/car_cfg20_eval_ebm3.py saved_model_vehicle20/checkpoint_epoch_80.pth

- Evaluate model on KITTI test:
- - $ cd ebms_3dod/3dod
- - $ python eval.py configs/car_cfg20_eval_ebm3_test.py saved_model_vehicle20/checkpoint_epoch_80.pth --out saved_model_vehicle20 (this creates 000000.txt - 007517.txt in ebms_3dod/3dod/saved_model_vehicle20)
- - To evaluate on KITTI test:
- - Download all 7518 files, mark all files and compress to a zip file
- - Upload the zip file to the KITTI evaluation server





***
***
***
## Pretrained model

- Model trained on KITTI train ($ python train.py configs/car_cfg20.py): https://drive.google.com/file/d/1hWKUZ4rx9h6Med3pI4A4wbHXEOey-8zI/view?usp=sharing
*
- Evaluate pretrained model on KITTI val:
- - Download the file checkpoint_epoch_80.pth from above and place in ebms_3dod/3dod/pretrained.
- - $ cd ebms_3dod/3dod
- - $ python eval.py configs/car_cfg20_eval_ebm3.py pretrained/checkpoint_epoch_80.pth
- - Expected output:
```
Car AP@0.90, 0.90, 0.90:
bbox AP:39.30, 31.42, 29.55
bev  AP:26.60, 22.03, 19.48
3d   AP:3.45, 2.74, 2.26
aos  AP:39.30, 31.39, 29.51
Car AP@0.85, 0.85, 0.85:
bbox AP:82.14, 67.97, 64.99
bev  AP:68.40, 58.62, 54.48
3d   AP:31.02, 23.91, 21.95
aos  AP:82.08, 67.89, 64.87
Car AP@0.80, 0.80, 0.80:
bbox AP:95.75, 86.92, 82.20
bev  AP:88.31, 80.06, 77.25
3d   AP:66.70, 54.32, 51.36
aos  AP:95.69, 86.79, 81.99
Car AP@0.75, 0.75, 0.75:
bbox AP:99.05, 93.37, 90.79
bev  AP:95.47, 87.54, 84.88
3d   AP:87.85, 74.96, 71.95
aos  AP:98.99, 93.18, 90.45
Car AP@0.70, 0.70, 0.70:
bbox AP:99.38, 96.16, 93.69
bev  AP:96.62, 92.93, 90.43
3d   AP:95.50, 86.83, 82.23
aos  AP:99.32, 95.89, 93.25
Car AP@0.50, 0.50, 0.50:
bbox AP:99.38, 96.16, 93.69
bev  AP:99.41, 96.35, 93.86
3d   AP:99.39, 96.29, 93.81
aos  AP:99.32, 95.89, 93.25
```
*
- Run pretrained model on KITTI test:
- - Download the file checkpoint_epoch_80.pth from above and place in ebms_3dod/3dod/pretrained.
- - $ cd ebms_3dod/3dod
- - $ python eval.py configs/car_cfg20_eval_ebm3_test.py pretrained/checkpoint_epoch_80.pth --out pretrained _(this creates 000000.txt - 007517.txt in ebms_3dod/3dod/pretrained)_
- - Download all 7518 files, mark all files and compress to a zip file
- - Upload the zip file to the KITTI evaluation server
- - Expexted output:
```
Benchmark	        Easy	Moderate	Hard
Car (Detection)	        96.81 %	93.54 %	88.33 %
Car (Orientation)	96.39 %	92.88 %	87.58 %
Car (3D Detection)	91.05 %	80.12 %	72.78 %
Car (Bird's Eye View)	95.64 %	89.86 %	84.56 %
````
