#ifndef OPTIMZER_H_20181120
#define OPTIMZER_H_20181120

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "map.h"
using namespace std;

namespace mono
{

class SemiDenseReprojectionErrorTerm{
  public:
    SemiDenseReprojectionErrorTerm(double measured_x, double measured_y):
      measured_x(measured_x), measured_y(measured_y){}
  
  template<typename T>
  bool operator()(const T *const pose,
                  const T *const point,
                  T *residual_ptr) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(pose, point ,p);
    p[0] += pose[3];
    p[1] += pose[4];
    p[2] += pose[5];

    //normalize coordinate
    T predicted_x = p[0] / p[2];
    T predicted_y = p[1] / p[2];

    //compute the residual
    residual_ptr[0] = predicted_x - T(measured_x);
    residual_ptr[1] = predicted_y - T(measured_y);

    return true;
  }

  static ceres::CostFunction *Create(
      const double &measured_x,
      const double &measured_y)
  {
    return new ceres::AutoDiffCostFunction<SemiDenseReprojectionErrorTerm, 2, 6, 3>(
      new SemiDenseReprojectionErrorTerm(measured_x, measured_y));
  }

  private:
    const double measured_x;
    const double measured_y;

};



struct ReprojectionErrorMonocular
{
    ReprojectionErrorMonocular(double x1, double y1)
        : x1(x1), y1(y1) {}

    template <typename T>
    bool operator()(const T *const pose,
                    const T *const point,
                    T *residual) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(pose, point, p);
        p[0] += pose[3];
        p[1] += pose[4];
        p[2] += pose[5];

        T nx = p[0] / p[2];
        T ny = p[1] / p[2];

        residual[0] = nx - T(x1);
        residual[1] = ny - T(y1);

        return true;
    }

    double x1;
    double y1;
};

class Optimizer
{
  private:
    SemiMap* world_;

  public:
    Optimizer(SemiMap* world):world_(world){};
    void SemiDenseOptimize();
};
} // namespace mono

#endif