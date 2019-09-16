#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 &
./main /home/daehyun/Data/eth_carpet/img/img%04d_0.png /home/daehyun/Data/eth_carpet/depth/img%04d_0.depth /home/daehyun/Data/eth_carpet/info/groundtruth.txt -logtostderr \
-canny_th_high 255
-canny_th_low 100
-motion_th 0.2
