#ifndef FRAME_H_2018_11_19
#define FRAME_H_2018_11_19

#include "common.h"
#include "util.h"
#include <opencv2/opencv.hpp>


namespace mono{

struct MyFrame{
    int frame_num;
    
    Mat34 gt_pose;
    
    cv::Mat depth_map;
    
    Mat3X normalized_pts;

    std::vector<uint> ids;

    std::vector<double> edgenesses;

    std::set<uint> inlier_ids;
    
    MyFrame(int frame_num, Mat34 gt_pose , cv::Mat depth_map, Mat3X normalized_pts, std::vector<uint>ids, std::vector<double>edgenesses,std::set<uint> inlier_ids):
        frame_num(frame_num), gt_pose(gt_pose), depth_map(depth_map), normalized_pts(normalized_pts), ids(ids),edgenesses(edgenesses), inlier_ids(inlier_ids){};

    MyFrame(){
        frame_num = -1; 
        gt_pose = Mat34::Identity();
        depth_map = cv::Mat();
        ids = std::vector<uint>();
        edgenesses = std::vector<double>();
        inlier_ids = std::set<uint>();
    };
};

}

#endif