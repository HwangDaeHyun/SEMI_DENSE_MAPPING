#ifndef EVALUATOR_H
#define EVALUATOR_H
#include "util.h"
#include "common.h"
#include "semi_tracker.h"
#include "semi_mapper.h"
namespace mono
{
void CalculateMeanEdgeness(const MyFrame &curr_frm, const std::vector<Edgel>& curr_feat){
    double edgeness_sum = 0.0;
    double cnt = 0.0;
    std::set<uint>::const_iterator citer;
    for (uint i =0; i < curr_feat.size() ; i++ ){
        const Edgel& ft = curr_feat[i];
        citer = curr_frm.inlier_ids.find(ft.id);
        if(citer != curr_frm.inlier_ids.end()){
            edgeness_sum += ft.edgeness;
            cnt +=1.0;
        }
    }
    std::cout << "edgeness mean : " <<  edgeness_sum / cnt <<std::endl;
}
void ComputeCorresopndence(const MyFrame &ref_frm, MyFrame &curr_frm, const Mat3 &intrinsic)
{
    const double fx = intrinsic(0, 0), fy = intrinsic(1, 1),
                 cx = intrinsic(0, 2), cy = intrinsic(1, 2);

    Mat3X new_pts1, new_pts2;
    vector<uint> matched_ids;

    const Mat3X &pts1 = ref_frm.normalized_pts;
    const Mat3X &pts2 = curr_frm.normalized_pts;

    const vector<uint> &ftids1 = ref_frm.ids;
    const vector<uint> &ftids2 = curr_frm.ids;

    Mat3 R = curr_frm.gt_pose.block(0,0,3,3);
    Vec3 t = curr_frm.gt_pose.block(0,3,3,1);
    // Find matching pair between 2 frame's features
    FindMatchedPoints(pts1, ftids1, pts2, ftids2, &new_pts1, &new_pts2, &matched_ids);
    uint inlier_cnt = 0;
    uint less_05 =0, less_01 =0, less_001 =0;
    
    for (uint i = 0; i < matched_ids.size(); i++)
    {
        Vec3 ref_np = new_pts1.col(i);
        Vec3 cur_np = new_pts2.col(i);

        double x = ref_np(0) * fx + cx;
        double y = ref_np(1) * fy + cy;

        ref_np = UnitVector(ref_np);
        if(x < 0 || y < 0 || x > 640 || y > 480)
            continue ;
        Vec3 p3d_tmp = ref_np * ref_frm.depth_map.at<double>(y, x);

        Vec3 gt_ip = R * p3d_tmp + t;
        gt_ip /= gt_ip(2);
        
        gt_ip(0) = gt_ip(0)*fx + cx;
        gt_ip(1) = gt_ip(1)*fy + cy;

        double cur_x  = cur_np(0)*fx + cx;
        double cur_y  = cur_np(1)*fy + cy;

        double px_dist = sqrt(pow(cur_x - gt_ip(0),2) + pow(cur_y -gt_ip(1), 2));
        if(px_dist <1.0 ){
            curr_frm.inlier_ids.insert(matched_ids[i]);
            inlier_cnt++;
        }        
        if(px_dist <0.5){
            //curr_frm.inlier_ids.insert(matched_ids[i]);
            less_05++;
        }
        if(px_dist <0.1){
            //curr_frm.inlier_ids.insert(matched_ids[i]);
            less_01++;
        }
        if(px_dist <0.01){
            //curr_frm.inlier_ids.insert(matched_ids[i]);
            less_001++;
        }
    }
    std::cout << "matching : " << matched_ids.size() << " inlier : " << inlier_cnt << " < 0.5px : " << less_05 << " < 0.1px : " << less_01 << " <0.01px : " << less_001 <<std::endl;
}

void ReprojectGTDepth(const MyFrame &ref_frm, MyFrame &curr_frm, const Mat3 &intrinsic)
{
    const double fx = intrinsic(0, 0), fy = intrinsic(1, 1),
                 cx = intrinsic(0, 2), cy = intrinsic(1, 2);

    const Mat3X &pts1 = ref_frm.normalized_pts;
    Mat3X &pts2 = curr_frm.normalized_pts;

    const vector<uint> &ftids1 = ref_frm.ids;
    const vector<uint> &ftids2 = curr_frm.ids;
    
    Mat3 R = curr_frm.gt_pose.block(0,0,3,3);
    Vec3 t = curr_frm.gt_pose.block(0,3,3,1);
    int m_cnt = 0;
    for (uint i = 0; i < ftids1.size(); ++i)
    {
        const uint ftid = ftids1[i];
        int idx = -1;
        for (uint j = 0; j < ftids2.size(); ++j)
        {
            if (ftid == ftids2[j])
                idx = j;
        }
        //const int idx = FindInSortedArray(ftids2, ftid);
        if (idx < 0)
            continue;
        Vec3 ref_np = pts1.col(i);
        
        double x = ref_np(0) * fx + cx;
        double y = ref_np(1) * fy + cy;
        if(x < 0 || y < 0 || x > 640 || y > 480)
            continue ;
        ref_np = UnitVector(ref_np);
        Vec3 p3d_tmp = ref_np * ref_frm.depth_map.at<double>(y, x);
        
        Vec3 gt_p = R * p3d_tmp + t;

        //gt_np(0) = (gt_np(0) - cx )/fx;
        //gt_np(1) = (gt_np(1) - cy )/fy;
        pts2.col(idx) = gt_p/gt_p(2);
        m_cnt++;
    }
    std::cout << "reprojected points : " << m_cnt << std::endl;
}
} // namespace mono
#endif