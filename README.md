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
