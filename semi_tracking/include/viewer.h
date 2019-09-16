#ifndef MONOSLAM_VIWER_H
#define MONOSLAM_VIWER_H

#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

#include <Eigen/Dense>
#include "common.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "image_file.h"
#include "image_util.h"
#include "semi_tracker.h"
#include "semi_mapper.h"

using namespace std;
using namespace mono;

namespace mono
{

template <class T>
inline string ToString(const T &v);

inline float *SetPointsToBuffer(const Mat &pts, float *buf);

inline float *SetPointsToBuffer(const Mat &pts, const int *col_idx, int num_pts,
                                float *buf);



class Viewer
{
private:
  int start_idx;
  int end_idx;
  const SemiMap *world;
  csio::OutputStream csio_out;

public:
  Viewer(int start, int end, const SemiMap *world) : start_idx(start), end_idx(end), world(world){};

  void Process(int idx,
               const MCImageGray8::ArrayType &image_array,
               const vector<Edgel> &feats,
               const vector<MyFrame> &frame_vec);
  void DrawGeometricOutput(vector<char> *geom_buf_ptr, const vector<MyFrame> &frame_vec);
};

} // namespace mono
#endif