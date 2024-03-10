[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FFARS&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /> 

# Fourier Adaptive Recognition System (FARS)

FARS with the integration of adversarial training and the Color Domain Aware Fourier Domain Adaptation (F2DA), the model ensures consistent feature extraction across diverse microscopy configurations.

The images showcase slides captured under both HCM and LCM microscopes at different magnifications. Detailed views on the right side of each image compare various annotation methods. It becomes evident that traditional bounding boxes often fail to tightly surround the target object, resulting in frequent overlaps with adjacent nuclei. However, pixel-level semantic segmentation labels offer a more precise location and distinct separation of each nucleus, including those instances where they overlap or touch. The figure also introduces modified bounding boxes, which are derived from the auto-generated segmentation labels. Notably, these modified boxes exhibit a tighter fit around the objects, thereby reducing unnecessary background inclusions.

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0010482524001392-gr1_lrg.jpg)

