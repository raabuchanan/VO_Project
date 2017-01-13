Visual Odometry project for Vision Algorithms for Mobile Robotics
======
#### by Alex Lieber, Carl Strömbeck, Russell Buchanan, Maximilian Enthoven
#### ETH Zurich / UZH, HS 2016

Basic visual odometry pipeline which uses Harris corner detectors to match keypoints between frames. RANSAC is performed on the p3p algorithm to determine pose and new landmarks are continually triangulated. Current implemetation is fairly robust but extremly slow (~1hz on a powerful enough laptop.)


Datasets
------
The pipeline been tested with the following datasets:
1. KITTI 00 http://www.cvlibs.net/datasets/kitti/
2. Malaga 07 http://www.mrpt.org/MalagaUrbanDataset

A Youtube video of the working pipeline at x10 speed is available here: https://youtu.be/0i2gmqZ2_mE
Pose estimates have been smoothed over the last 10 frames.

![](kitti1000.png?raw=true)
