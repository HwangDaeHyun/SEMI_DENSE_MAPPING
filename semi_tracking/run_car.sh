#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0  &
gdb --args ./main /home/daehyun/Data/eth_car/data/img/img%04d_0.png /home/daehyun/Data/eth_car/data/depth/img%04d_0.depth /home/daehyun/Data/eth_car/info/groundtruth.txt -logtostderr \
-canny_th_high 150 \
-canny_th_low 100 \
-motion_th 0.25 \
-klt_num_levels 3 \
-display_cam_size 0.5 \
-display_point_size 1.5 \
-angle_th 0 \
-start 2 \
-end 2500 \
