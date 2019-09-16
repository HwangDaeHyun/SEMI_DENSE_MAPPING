//author : daehyun hwang (ghkdeoek@gmail.com)
#include "semi_mapper.h"
#define PI 3.14159265359

DEFINE_double(angle_th,3, "index of end frame");
namespace mono
{
void SemiMapper::FindMatchedPointsMutliView(std::vector<Mat34 *> *poses,
                                            std::vector<Mat3X> *new_pts,
                                            std::vector<uint> *matched_ids,
                                            std::vector<double>& matched_edgeness)
{
    int cur_frm_idx = this->world->all_frames.size() - 1;
    int end_frm_idx = cur_frm_idx - this->mapping_interval+1;
    assert(end_frm_idx >= 0);

    std::vector<MyFrame> &all_frame = this->world->all_frames;
    const MyFrame &cur_frm = all_frame[cur_frm_idx];

    //make pose set
    for (int ref_idx = cur_frm_idx; ref_idx >= end_frm_idx; ref_idx--)
        poses->push_back(&all_frame[ref_idx].gt_pose);
    
    new_pts->resize(this->mapping_interval);
    
    for(uint i =0 ; i < new_pts->size() ; i++){
        new_pts->at(i).resize(3, cur_frm.normalized_pts.cols());
    }

    vector<uint> idx_set;
    for (uint i = 0; i < cur_frm.ids.size(); i++)
    {
        bool all_found = true;
        const uint ft_id = cur_frm.ids[i];
        const double ft_edgeness = cur_frm.edgenesses[i];
        idx_set.clear();
        idx_set.push_back(i);
        //search all ids
        for (int ref_idx = cur_frm_idx - 1; ref_idx >= end_frm_idx; ref_idx--)
        {
            std::vector<uint>::iterator iter;
            iter = find(all_frame[ref_idx].ids.begin(), all_frame[ref_idx].ids.end(), ft_id);
            if (iter == all_frame[ref_idx].ids.end())
            {
                all_found = false;
                break;
            }
            else
            {
                idx_set.push_back(iter - all_frame[ref_idx].ids.begin());
            }
        }
        
        if (all_found)
        {
            const int col_idx = matched_ids->size();
            for (int j = 0, ref_idx = cur_frm_idx; ref_idx >= end_frm_idx; ref_idx--, j++)
            {
                new_pts->at(j).col(col_idx) = all_frame[ref_idx].normalized_pts.col(idx_set[j]);
                //std::cout << new_pts->at(j).col(col_idx) << std::endl;
            }
            matched_ids->push_back(ft_id);
            matched_edgeness.push_back(ft_edgeness);
        }
    }
}

void SemiMapper::MultiViewTriangulate(const vector<Mat34 *>*poses,
                                      const vector<Mat3X> *pts,
                                      Eigen::ArrayXXd *pts_4d)
{
    //two vector size are must same, and more than 0
    assert(poses->size() == pts->size() && poses->size() > 0);

    Mat A(2*poses->size(), 4);

    for (uint i = 0; i < pts->at(0).cols(); i++)
    {
        for (uint j = 0; j < poses->size(); j++)
        {
            double x, y;
            x = (pts->at(j))(0, i);
            y = (pts->at(j))(1, i);
            for (uint k = 0; k < 4; ++k)
            {
                A(j * 2 + 0, k) = x * (*poses->at(j))(2, k) - (*poses->at(j))(0, k);
                A(j * 2 + 1, k) = y * (*poses->at(j))(2, k) - (*poses->at(j))(1, k);
            }
        }
        Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
        pts_4d->col(i) = svd.matrixV().col(3);
    }
    pts_4d->rowwise() *= 1. / pts_4d->row(3);
}
void SemiMapper::ProcessMapping(uint m_interval)
{
    //Set Mapping Interval
    this->SetMappingInterval(m_interval);

    std::vector<Mat34*> poses;
    std::vector<Mat3X> new_pts;
    std::vector<uint> matched_ids;
    std::vector<double> matched_edgeness;
    this->FindMatchedPointsMutliView(&poses, &new_pts, &matched_ids, matched_edgeness);

    Eigen::ArrayXXd pts_homo(4, matched_ids.size());
    this->MultiViewTriangulate(&poses, &new_pts, &pts_homo);
    pts_homo.rowwise() /= pts_homo.row(3);

    const MyFrame &curr_frm = this->world->all_frames[this->world->all_frames.size()-1];
    Mat3X curview_pts = curr_frm.gt_pose * pts_homo.matrix();

    std::set<uint>::const_iterator citer;

    for (uint j = 0; j < matched_ids.size(); ++j)
    {
        bool find = false;
        const int ftid = matched_ids[j];
        const double edgeness = matched_edgeness[j];
        map<int, MapPoint>::const_iterator it = this->world->semi_map.find(ftid);
        Vec3 pos;
        // Add new points
        citer = curr_frm.inlier_ids.find(ftid);
        if(citer == curr_frm.inlier_ids.end())
         continue;
        if (it == this->world->semi_map.end())
        {
            //TODO : Running Mean or Running Variance
            pos = pts_homo.col(j).block(0, 0, 3, 1);
            const Vec3 rel_pos = curview_pts.col(j);
            
            //delete z<0
            if (rel_pos(2) <= 0)
                continue;
            find = true;
        }
        else
        {
            pos = pts_homo.col(j).block(0, 0, 3, 1);
            const Vec3 rel_pos = curview_pts.col(j);
            //delete z < 0
            if (rel_pos(2) <= 0)
                continue;
            MapPoint map_pts(ftid, pos,edgeness, curr_frm.frame_num);
            this->world->semi_map[ftid] = map_pts;
            find = false;
        }
        if (find)
        {
            MapPoint map_pts(ftid, pos,edgeness ,curr_frm.frame_num);
            this->world->semi_map.insert({ftid, map_pts});
        }
    }
}
void TriangulatePoints(const Mat34 &pose0, const Mat34 &pose1,
                       const Mat3X &pts0, const Mat3X &pts1,
                       Eigen::ArrayXXd *pts_4d)
{
    int n = pts0.cols();
    Mat A(4, 4);
    const Mat3X *pts[2] = {&pts0, &pts1};
    const Mat34 *pose[2] = {&pose0, &pose1};

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double x, y;
            x = (*(pts[j]))(0, i);
            y = (*(pts[j]))(1, i);
            for (int k = 0; k < 4; ++k)
            {
                A(j * 2 + 0, k) = x * (*pose[j])(2, k) - (*pose[j])(0, k);
                A(j * 2 + 1, k) = y * (*pose[j])(2, k) - (*pose[j])(1, k);
            }
        }
        Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
        pts_4d->col(i) = svd.matrixV().col(3);
    }
    pts_4d->rowwise() *= 1. / pts_4d->row(3);
}

void MakeNormalizedPoints(const std::vector<Edgel> &pts, Mat3X *normalized_pts,
                          const Mat3 &intrinsic,
                          const Vec2 &radial_distortion,
                          const Vec3 &tangential_distortion)
{
    const double fx = intrinsic(0, 0), fy = intrinsic(1, 1), cx = intrinsic(0, 2), cy = intrinsic(1, 2);
    const double k0 = radial_distortion(0), k1 = radial_distortion(1);
    const double k2 = tangential_distortion(0);
    const double k3 = tangential_distortion(1);
    const double k4 = tangential_distortion(2);

    normalized_pts->resize(3, pts.size());
    for (uint idx = 0; idx < pts.size(); idx++)
    {
        double nx = (pts[idx].pos(0) - cx) / fx;
        double ny = (pts[idx].pos(1) - cy) / fy;
        double nx_out, ny_out;
        if (k1 != 0.0 || k2 != 0.0)
        {
            double x = nx, y = ny;
            for (int i = 0; i < 10; i++)
            {
                const double x2 = pow(x, 2), y2 = pow(y, 2), xy = 2 * x * y, r2 = x2 + y2;
                const double rad = 1 + r2 * (k0 + r2 * (k1 + r2 * k4));
                const double ux = (nx - (xy * k2 + (r2 + 2 * x2) * k3)) / rad;
                const double uy = (ny - ((r2 + 2 * y2) * k2 + xy * k3)) / rad;
                const double dx = x - ux, dy = y - uy;
                x = ux, y = uy;
                if (pow(dx, 2) + pow(dy, 2) < 1e-9)
                    break;
            }
            nx = x, ny = y;
        }
        nx_out = nx, ny_out = ny;
        normalized_pts->col(idx) << nx_out, ny_out, 1;
    }
}

void FindMatchedPoints(const Mat3X &pts1, const vector<uint> &ftids1,
                       const Mat3X &pts2, const vector<uint> &ftids2,
                       Mat3X *new_pts1, Mat3X *new_pts2,
                       vector<uint> *matched_ids)
{
    matched_ids->clear();
    const int n = pts1.cols() > pts2.cols() ? pts1.cols() : pts2.cols();
    new_pts1->resize(3, n);
    new_pts2->resize(3, n);
    matched_ids->reserve(n);

    //Find correspondance points on each 2 frames
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
        const Vec3 &p1 = pts1.col(i);
        const Vec3 &p2 = pts2.col(idx);
        const int col_idx = matched_ids->size();
        new_pts1->col(col_idx) = p1;
        new_pts2->col(col_idx) = p2;
        matched_ids->push_back(ftid);
    }
    new_pts1->conservativeResize(3, matched_ids->size());
    new_pts2->conservativeResize(3, matched_ids->size());
}

double UpdateWorldPoints(MyFrame &prev_frm, MyFrame &curr_frm, SemiMap *world_ )
{
    double ret_ratio = 0.0;
    int enroll_cnt =0;
    Mat3X new_pts1, new_pts2;
    vector<uint> matched_ids;

    const Mat3X &pts1 = prev_frm.normalized_pts;
    const Mat3X &pts2 = curr_frm.normalized_pts;

    const vector<uint> &ftids1 = prev_frm.ids;
    const vector<uint> &ftids2 = curr_frm.ids;

    // Find correspondance between 2 frame's features
    FindMatchedPoints(pts1, ftids1, pts2, ftids2, &new_pts1, &new_pts2, &matched_ids);
    //std::cout << "matched : " << matched_ids.size() << std::endl;
    Mat34 pose_prev = prev_frm.gt_pose;
    Mat34 pose_cur = curr_frm.gt_pose;

    std::set<uint>::const_iterator citer;

    // Triangulation
    Eigen::ArrayXXd pts_4d(4, matched_ids.size());
    TriangulatePoints(pose_prev, pose_cur, new_pts1, new_pts2, &pts_4d);
    pts_4d.rowwise() /= pts_4d.row(3);

    Mat3X pts3d = pose_cur * pts_4d.matrix();

    for (uint j = 0; j < matched_ids.size(); ++j)
    {
        bool find = false;
        const int ftid = matched_ids[j];
        map<int, MapPoint>::const_iterator it = world_->semi_map.find(ftid);
        Vec3 pos;
        // Add new points
        citer = curr_frm.inlier_ids.find(ftid);
        
        if (it == world_->semi_map.end())
        {
            pos = pts_4d.col(j).block(0, 0, 3, 1);
            const Vec3 rel_pos = pts3d.col(j);
            //delete z<0
            double theta = atan2((new_pts1.col(j).cross(new_pts2.col(j))).norm(),new_pts1.col(j).dot(new_pts2.col(j)));
            //std::cout << theta*180/PI << std::endl;
            if(theta*180/PI < FLAGS_angle_th)
                continue;
            
            if (rel_pos(2) <= 0)
                continue;
            find = true;
        }
        else
        {
            pos = pts_4d.col(j).block(0, 0, 3, 1);
            
            const Vec3 rel_pos = pts3d.col(j);
            //delete z < 0
            if (rel_pos(2) <= 0)
                continue;
            find = false;
        }
        if (find)
        {
            enroll_cnt++;   
            MapPoint map_pts(ftid, pos, 0.0,curr_frm.frame_num);
            world_->semi_map.insert(std::pair<int , MapPoint>(ftid, map_pts));
        }
    }
    ret_ratio = (double)enroll_cnt/(double)matched_ids.size();
    return ret_ratio;
}

} // namespace mono
