# Semi Dense Tracking and Mapping (IPIU 2019)
**Authors:** [DaeHyun Hwang] 

<img src="https://user-images.githubusercontent.com/16742591/64946353-a4f56c00-d8ad-11e9-9c11-e1d925315c36.png" 
alt="SLAM" width="212" height="199" border="10" />

<img src="https://user-images.githubusercontent.com/16742591/64947215-7c6e7180-d8af-11e9-8639-906644fd76a1.png" 
alt="SLAM" width="344" height="278" border="10" /></a>

# Prerequisites
We have tested the library in **Ubuntu 16.04 , macOS high sierra**

## CSIO
We use [CSIO](https://github.com/jwlim/csio) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/jwlim/csio.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at least 3.0**.

## Eigen3
Required by ceres (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.0.0**.

##ceres (Included in Thirdparty folder)
We use modified versions of the [ceres](https://github.com/ceres-solver/ceres-solver) library to perform non-linear optimizations.


# Build

setp 1. CSIO Build
 - cd csio
 - mkdir build
 - cmake ../
 - make

step 2. semi_tracking build
 - mkdir build
 - cd build
 - cmake ../
 - make

step 3. Set dataset path
 - cp run.sh ./build/ && cd build
 - vi run.sh
 - Data_DIR : set dataset path 

step 4. Make pipeline for the CISO and run excute file
 - mkfifo side2.csio
 - sh your_runscript.sh
```








