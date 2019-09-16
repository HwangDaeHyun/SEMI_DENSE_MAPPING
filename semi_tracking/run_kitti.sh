#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 &
./main /home/cvlab/test_set/06/image_0/%06d.png -klt_max_features 500 \
-klt_redetect_thr 300 \
-klt_min_cornerness 1 \
-klt_num_levels 6 \

