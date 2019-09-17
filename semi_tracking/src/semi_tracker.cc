// semi_tracker.cc
// author: DaeHyun Hwang(ghkdeoek@gmail.com)
//

#include <math.h>
#include <stdarg.h>

#include <map>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "semi_tracker.h"

// /extern mono::ProfileDBType pdb_;

using namespace std;

DEFINE_int32(klt_num_levels, 4, "The number of image pyramid levels in KLT Tracker.");
DEFINE_double(klt_sigma, 1.0, "Gaussian sigma for KLT tracker.");
DEFINE_double(klt_min_cornerness, 10, "The mininum cornerness response.");
DEFINE_int32(klt_max_features, 100, "The maximum number of features to track.");
DEFINE_int32(klt_redetect_thr, 50, "The threshold to redetect features.");
DEFINE_int32(klt_num_loop,7, "The number of loops in each iteration.");
DEFINE_int32(klt_detect_level, 1, "The pyramid level to detect features.");
DEFINE_double(tracker_min_edgeness, 2, "minimum Edgeness response.");
DEFINE_int32(canny_th_high ,125 ,"max threshold for canny edge");
DEFINE_int32(canny_th_low , 100 ,"low threshold for canny edge");

namespace mono{

typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
typedef Eigen::ArrayXXf ArrayF;

template <class T>
inline string ToString(const T &v)
{
  stringstream ss;
  ss << v;
  return ss.str();
}

inline bool operator<(const Edgel &ft1, const Edgel &ft2)
{
  return ft1.score < ft2.score;
}

template <typename T, int d, int e>
inline Eigen::Matrix<T, d + e, 1> Concat(const Eigen::Matrix<T, d, 1> &v1,
                                         const Eigen::Matrix<T, e, 1> &v2)
{
  Eigen::Matrix<T, d + e, 1> ret;
  ret << v1, v2;
  return ret;
}
void EdgeTracker::FindCanny(const ImagePyramid::Level &level, int level_idx,
               const ArrayF &gaussian_kernel, double min_edgeness,
               vector<Edgel> *feats)
{
  const int rad = gaussian_kernel.rows() / 2;
  const int width = level.imgf.rows(), height = level.imgf.cols();
  ArrayF edgeness(width, height);

  cv::Mat edge_img;
  Eigen::MatrixXf tmp_eigen = level.imgf.matrix();
  cv::eigen2cv(tmp_eigen, edge_img);
  edge_img.convertTo(edge_img, CV_8UC1);  

  cv::Canny(edge_img, edge_img, FLAGS_canny_th_low, FLAGS_canny_th_high, 3);
  const double kExistingFeature = 1e9;
  const double level_scale = (1L << level_idx);
  for (unsigned int i = 0; i < feats->size(); ++i)
  {
    Edgel &ft = feats->at(i);
    int x = ft.pos[0] / level_scale, y = ft.pos[1] / level_scale;
    ft.score = edgeness(x, y);
    edgeness(x, y) = kExistingFeature;
  }
  int exist_cnt = 0;
  //std::cout << edge_img.size() << std::endl;
  int margin = rad * 2 + 1;
  for(int y = margin; y < height - margin; ++y)
  {
    for(int x = margin; x < width - margin; ++x)
    {
      if(edgeness(x,y) == kExistingFeature){
        exist_cnt++;
        continue;
      }

      if(edge_img.at<unsigned char>(x,y) > 0 && edgeness(x,y) != kExistingFeature){
        Eigen::Vector2d pos(x * level_scale, y * level_scale);
        feats->push_back(Edgel(this->next_id_++, level_idx, pos, 0, Vec3(0, 0, 0), 0));
      }
    }
  }
  //std::cout << "exist cnt : " << exist_cnt  << std::endl;
}

void FindEdgels(const ImagePyramid::Level &level, int level_idx,
                const ArrayF &gaussian_kernel, double min_edgeness,
                vector<Edgel> *feats, Mat3 fMat)
{
  // Compute cornerness.
  const int rad = gaussian_kernel.rows() / 2;
  const int width = level.imgf.rows(), height = level.imgf.cols();
  const int win_size = (2 * rad + 1);
  ArrayF edgeness(width, height);
  ArrayF img_x2 = level.imgx * level.imgx;
  ArrayF img_xy = level.imgx * level.imgy;
  ArrayF img_y2 = level.imgy * level.imgy;
  ArrayF tmp(width, height);
  Convolve(img_x2, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_x2);
  Convolve(img_xy, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_xy);
  Convolve(img_y2, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_y2);
  const int fbit = 10;
  const int tan225 = 0.559;
  const int tan675 = 22.5882;
  //Compute Edgenes
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const double ix2 = img_x2(x, y), ixy = img_xy(x, y), iy2 = img_y2(x, y);
      const double p1 = ix2 + iy2;
      const double p2 = sqrt(4 * ixy * ixy + (ix2 - iy2) * (ix2 - iy2));

      //edgeness(x, y) = 0.5 * max(p1 + p2, p1 - p2) + 1e-6 * rand() / RAND_MAX;
      //edgeness(x, y) = pow((epl(0) / mag) * level.imgx(x, y) + (epl(1) / mag) * level.imgy(x, y), 2);
      edgeness(x, y) = sqrt(pow(level.imgx(x, y), 2) + pow(level.imgy(x, y), 2));
    }
  }

  // Mark existing features.
  const double kExistingFeature = 1e9;
  const double level_scale = (1L << level_idx);
  for (unsigned int i = 0; i < feats->size(); ++i)
  {
    Edgel &ft = feats->at(i);
    int x = ft.pos[0] / level_scale, y = ft.pos[1] / level_scale;
    ft.score = edgeness(x, y);
    edgeness(x, y) = kExistingFeature;
  }
  int margin = rad * 2 + 1;
  for (int y = margin; y < height - margin; ++y)
  {
    for (int x = margin; x < width - margin; ++x)
    {
      const double edgeness_x_y = edgeness(x, y);
      if (edgeness_x_y == kExistingFeature ||
          edgeness_x_y < min_edgeness)
        continue;
      bool local_max = true;


      if (local_max )
      {
        Eigen::Vector2d pos(x * level_scale, y * level_scale);
        feats->push_back(Edgel(-1, level_idx, pos, 0, Vec3(0, 0, 0), edgeness_x_y));
      }
    }
  }
}


size_t FilterDuplicateTracks(int rad, vector<Edgel> *feats)
{
  rad = 0;
  multimap<float, int> ypos;

  for (unsigned int i = 0; i < feats->size(); ++i)
  {
    const Eigen::Vector2d &p = feats->at(i).pos;
    if (p[0] >= 0 && p[1] >= 0)
      ypos.insert(make_pair(p[1], i));
  }
  vector<Edgel> tmp_feats;
  tmp_feats.reserve(feats->size());
  for (unsigned int i = 0; i < feats->size(); ++i)
  {
    const Edgel &ft = feats->at(i);
    const int ft_x = ft.pos[0], ft_y = ft.pos[1];
    if (ft_x < 0 || ft_y < 0||ft.score > 7){

      continue;
    }
    multimap<float, int>::const_iterator it, it_lb, it_ub;
    it_lb = ypos.lower_bound(ft_y - rad);
    it_ub = ypos.upper_bound(ft_y + rad);
    bool duplicate = false;
    for (it = it_lb; it != it_ub && !duplicate; ++it)
    {
      const Edgel &ft2 = feats->at(it->second);
      duplicate = (ft_x - rad < ft2.pos[0] && ft2.pos[0] < ft_x + rad &&
                   ft.score < ft2.score);
    }
    if (!duplicate)
    {
      tmp_feats.push_back(ft);
    }
  }
  //feats->clear();
  feats->swap(tmp_feats);
  return feats->size();
}

bool EdgeTracker::ExtractHighGradientPoints(const ImagePyramid &pyr, const ArrayF &gaussian_kernel, double min_grad, vector<Edgel> *feats, Mat3 &fMat)
{
  const int org_size = feats->size();
  //const int rad = gaussian_kernel.rows() / 2;

  const ImagePyramid::Level &level = pyr[0];
  //FindEdgels(level, 0, gaussian_kernel, FLAGS_tracker_min_edgeness, feats, fMat);
  FindCanny(level, 0, gaussian_kernel, FLAGS_tracker_min_edgeness, feats);
  for (unsigned int i = org_size; i < feats->size(); ++i)
  {
    Edgel &ft = feats->at(i);
    ft.score = 0.0;
  }
  return true;
}

double Distance(double x0, double y0, double x1, double y1)
{
  const double dx = x0 - x1, dy = y0 - y1;
  return sqrt(dx * dx + dy * dy);
}

double Distance(const Vec3 &pt1, const Vec3 &pt2)
{
  return Distance(pt1(0), pt1(1), pt2(0), pt2(1));
}

bool EdgeTracker::KLTTrackFeaturesUsingExtrinsic(const ImagePyramid &prev, const ImagePyramid &cur,
                                    const ArrayF &gaussian_kernel, const int num_loop,
                                    vector<Edgel> *feats, const Mat3 &fMat)
{

  if (feats->size() <= 0)
    return true;
  // Track features from the top level to the bottom.
  const int rad = gaussian_kernel.rows() / 2;
  const int win_size = (2 * rad + 1);
  ArrayF patch_dx(win_size, win_size), patch_dy(win_size, win_size);
  ArrayF patch_dt(win_size, win_size);
  ArrayF patch0(win_size, win_size);
  ArrayF patch1_dx(win_size, win_size);
  ArrayF patch1_dy(win_size, win_size);
  ArrayF patch1_dt(win_size, win_size);
  ArrayF patch_tmp(win_size, win_size);

  //epipole line jacobian amtrix
  ArrayF patch_jx(win_size, win_size);
  ArrayF patch_jy(win_size, win_size);
  ArrayF patch_ml2ix_pl1iy(win_size, win_size);
  ArrayF patch_pl1ix_pl2iy(win_size, win_size);
  for (unsigned int i = 0; i < feats->size(); ++i)
  {
    const int num_levels = cur.num_levels();
    const double pow2level = (1L << num_levels);
    Edgel &ft = feats->at(i);
    double x = ft.pos[0] / pow2level;
    double y = ft.pos[1] / pow2level;
    double x0 = x, y0 = y;
    double score = 0.0;
    double edgeness = 0.0;
    Vec3 epl = ComputeEpipolarline(fMat, Vec2(ft.pos[0], ft.pos[1]));
    ft.epl << epl(0), epl(1), epl(2);

    for (int l = num_levels - 1; l >= 0; --l)
    {
      x *= 2, y *= 2, x0 *= 2, y0 *= 2;
      double s = pow(2.0, l);
      Vec2 op = OrthogonalProjectionOfPointOntoEpipolarLine(Vec2(x * s, y * s), ft.epl);
      x = op(0) / s;
      y = op(1) / s;
      if (l < ft.level)
        continue;
      const ImagePyramid::Level &level_prev = prev[l];
      const ImagePyramid::Level &level_cur = cur[l];
      const int width = level_cur.imgf.rows(), height = level_cur.imgf.cols();
      const int x_max = width - rad - 1, y_max = height - rad - 1;
      //gradient x
      Interp2Patch(level_prev.imgx, x0, y0, &patch1_dx);
      //gradient y
      Interp2Patch(level_prev.imgy, x0, y0, &patch1_dy);
      //img patch
      Interp2Patch(level_prev.imgf, x0, y0, &patch1_dt);
      patch1_dt -= patch1_dt.sum() / patch1_dt.size();

      for (int loop = 0; loop < num_loop; ++loop)
      {
        if (!(rad <= x && rad <= y && x < x_max && y < y_max))
        {
          x = -1.f, y = -1.f; // Set invalid coordinates.
          break;
        }
        //double e[2] = {0, 0}, Z[3] = {0, 0, 0};
        score = 0.0;
        edgeness = 0.0;
        //std::cout << patch_dx <<std::endl;
        Interp2Patch(level_cur.imgx, x, y, &patch0);
        patch_dx = patch0;
        Interp2Patch(level_cur.imgy, x, y, &patch0);
        patch_dy = patch0;
        Interp2Patch(level_cur.imgf, x, y, &patch0);
        patch0 -= patch0.sum() / patch0.size();

        //make jacobian matrix
        for (int rx = -rad; rx <= rad; rx++)
        {
          for (int ry = -rad; ry <= rad; ry++)
          {
            double jx = rx + x;
            double jy = ry + y;
            Vec3 ep_line = ComputeEpipolarline(fMat, {jx * s, jy * s});
            patch_jx(rad + rx, rad + ry) = ep_line(0) / sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));
            patch_jy(rad + rx, rad + ry) = ep_line(1) / sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));
          }
        }
        //multiply jacobian and gradient
        for (int rx = 0; rx < 2 * rad + 1; rx++)
        {
          for (int ry = 0; ry < 2 * rad + 1; ry++)
          {
            patch_ml2ix_pl1iy(rx, ry) = -patch_jy(rx, ry) * patch_dx(rx, ry) + patch_jx(rx, ry) * patch_dy(rx, ry);
            patch_pl1ix_pl2iy(rx, ry) = patch_jx(rx, ry) * patch_dx(rx, ry) + patch_jy(rx, ry) * patch_dy(rx, ry);
          }
        }
        patch_dt = patch0 - patch1_dt;

        score += patch_dt.abs().sum() / patch_dt.size();
        //edgeness += patch_ml2ix_pl1iy.square().sum() / patch_ml2ix_pl1iy.size();
        const double coef = 1.0 / patch_ml2ix_pl1iy.square().sum();
        const double lamda = coef * ((patch_ml2ix_pl1iy * patch_dt).sum());

        Vec3 ep_line = ComputeEpipolarline(fMat, {x0 * s, y0 * s});

        double dx = -lamda * ep_line(1) / sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));
        double dy = lamda * ep_line(0) / sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));

        x += dx;
        y += dy;
      }
      if (!(rad <= x && rad <= y && x < x_max && y < y_max))
      {
        x = -1.f, y = -1.f;
        break;
      }
    }
    ft.pos << x, y;
    ft.score = score;
    //ft.edgeness = edgeness;
  }
  return true;
}

EdgeTracker::EdgeTracker()
    : frame_no_(0), next_id_(0), detect_count_(3), isInit(false), isRedect(false),
      pyramid_builder_(FLAGS_klt_num_levels, FLAGS_klt_sigma)
{
}

bool EdgeTracker::Setup(const ArrayXXu8 &image, Mat3 &intrinsic, const Mat34 &prev_pose, const Mat34 &curr_pose)
{
  if (pyramid_builder_.Build(image, &prev_) == false)
    return false;

  this->features_.clear();
  this->intrinsic = intrinsic;

  Mat3 fMat = ComputeFundamentalMatrixFromTwoPoses(prev_pose, curr_pose, intrinsic);
  //# extract high gradient points (edge)
  ExtractHighGradientPoints(prev_, pyramid_builder_.gaussian_1d_kernel(), 0.5, &features_, fMat);

  this->isInit = RelativeTransform(prev_pose, curr_pose).norm() > 1 ? true : false;
  return true;
}
               
bool EdgeTracker::Process(const ArrayXXu8 &image, const Mat34 &prev_pose, const Mat34 &curr_pose)
{
  ++frame_no_;

  ImagePyramid cur;
  if (pyramid_builder_.Build(image, &cur) == false)
    return false;

  const ArrayF &gaussian_kernel = pyramid_builder_.gaussian_1d_kernel();

  //Compute Fundamental Matrix with Relative Motion (GT pose ver.)
  Mat3 fMat = ComputeFundamentalMatrixFromTwoPoses(prev_pose, curr_pose, this->intrinsic);

  KLTTrackFeaturesUsingExtrinsic(prev_, cur, gaussian_kernel, FLAGS_klt_num_loop, &features_, fMat);
  
  FilterDuplicateTracks(0, &features_);
  const int idx0 = features_.size();
  //ExtractEdgedFeatures(image,&features_);
  if (this->isRedect)
  {
    ExtractHighGradientPoints(cur, pyramid_builder_.gaussian_1d_kernel(), 0.5, &features_, fMat);
    
    this->isRedect = false;
  }
  prev_.Swap(&cur);
  return true;
}

} // namespace mono
