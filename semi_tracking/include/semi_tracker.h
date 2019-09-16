// klt_tracker.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// copied by : Daehyun Hwang (ghkdeoek@gmail.com)
//
#ifndef _SEMI_TRACKER_H_
#define _SEMI_TRACKER_H_

#include <fstream>
#include <map>
#include <vector>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>
#include <Eigen/Dense>
#include "image_pyramid.h"
#include "common.h"
#include "util.h"
#include "frame.h"
namespace mono {


struct Edgel {
  uint id, level;
  Eigen::Vector2d pos;
  double score;
  Vec3 epl;
  double edgeness;

  Edgel() : id(-1), level(0), pos({0,0}), score(0.0),epl({0,0,0}),edgeness(0.0) {}
  Edgel(const Edgel& ft)
      : id(ft.id), level(ft.level), pos(ft.pos), score(ft.score),epl(ft.epl),edgeness(ft.edgeness) {}
  Edgel(int id, int level, const Eigen::Vector2d& pos, double score, Vec3 epl,double edgeness)
      : id(id), level(level), pos(pos), score(score), epl(epl),edgeness(edgeness){}

  void Set(int id_, int level_, const Eigen::Vector2d& pos_, double score_, double edgeness_) {
    id = id_, level = level_, pos = pos_, score = score_, edgeness= edgeness_;
  }
};

class EdgeTracker {

 public:
 
  typedef Eigen::ArrayXXf ArrayF;
  typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
  typedef Eigen::Vector2f Vec2f;
  typedef Eigen::VectorXf VecXf;

  EdgeTracker();
  bool Setup(const ArrayXXu8 &image, Mat3 &intrinsic,  const Mat34& prev_pose ,const Mat34& curr_pose);
  
  bool Process(const ArrayXXu8 &image, const Mat34 &prev_pose, const Mat34 &cur_pose);
  void Cleanup();

  void SetRedect(bool isRedect){ this-> isRedect = isRedect;}
  bool IsSetup() const { return prev_.num_levels() > 0; }
  bool isInitialized() const{ return isInit ;}
  int frame_no() const { return frame_no_; }

  const std::vector<Edgel>& features() const { return features_; }

  void RemoveFeatures(const std::vector<int>& feature_idx);
  const ImagePyramid& image_pyramid() const { return prev_; }

  bool KLTTrackFeaturesUsingExtrinsic(const ImagePyramid &prev, const ImagePyramid &cur,
                                    const ArrayF &gaussian_kernel, const int num_loop,
                                    std::vector<Edgel> *feats, const Mat3 &fMat);
  
  bool ExtractHighGradientPoints(const ImagePyramid &pyr, const ArrayF &gaussian_kernel, double min_grad, std::vector<Edgel> *feats, Mat3 &fMat);
  void FindCanny(const ImagePyramid::Level &level, int level_idx,
               const ArrayF &gaussian_kernel, double min_edgeness,
               std::vector<Edgel> *feats);
 private:
  Mat3 intrinsic;
  int frame_no_;
  int next_id_;
  
  int detect_count_;
  std::vector<Edgel> features_;
  
  bool isInit;
  bool isRedect;
  ImagePyramid prev_;
  ImagePyramidBuilder pyramid_builder_;
};
}
#endif  // _RVSLAM_KLT_TRACKER_H_
