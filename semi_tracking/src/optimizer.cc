#include "optimizer.h"

DEFINE_int32(optimze_windows, 10, "Number of keyframes to keep.");
DEFINE_double(ceres_huber_loss_sigma,1.0,
              "The sigma parameter of Huber loss function.");
namespace mono
{
void Optimizer::SemiDenseOptimize()
{
  std::cout << "key_frames size  : "<< world_->key_frames.size() << std::endl;
  if (world_->key_frames.size() < 3 || world_->semi_map.size() < 1)
    return;
  std::cout << "op" << std::endl;
  ceres::Problem problem;
  std::vector<Vec6> pose_vec;
  uint num_iter = FLAGS_optimze_windows > world_->key_frames.size() ? world_->key_frames.size() : FLAGS_optimze_windows;
  for (uint f_idx = 0; f_idx < num_iter; ++f_idx)
  {
    MyFrame &frm = world_->key_frames[world_->key_frames.size() - f_idx-1];
    //to posevector
    Vec6 pose;
    Mat3 rot = frm.gt_pose.block(0,0,3,3);
    ceres::RotationMatrixToAngleAxis(rot.data(), pose.data());
    pose.segment(3, 3) = frm.gt_pose.block(0, 3, 3, 1);
    pose_vec.push_back(pose);

    
    int cnt = 0;
    for (uint i = 0; i < frm.ids.size(); i++)
    {
      map<int, MapPoint>::iterator it = world_->semi_map.find(frm.ids[i]);

      if (it == world_->semi_map.end())
      {
        //LOG(FATAL) << "ftid " << ftid << " not found.";
        continue;
      }
      MapPoint& mpt = it->second;
      Vec3* pts3d = &mpt.pt3d;

      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<ReprojectionErrorMonocular, 2, 6, 3>(
              new ReprojectionErrorMonocular(
                  frm.normalized_pts(0, i), frm.normalized_pts(1, i)));

      ceres::LossFunction *loss_function =
          new ceres::HuberLoss(FLAGS_ceres_huber_loss_sigma);

      problem.AddResidualBlock(cost_function, loss_function,
                               pose_vec[pose_vec.size()-1].data(), pts3d->data());
      
      
    }
    problem.SetParameterBlockConstant(pose_vec[pose_vec.size()-1].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 200;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //LOG(INFO) << summary.BriefReport();
  std::cout << summary.FullReport() << std::endl;
  //lba_ftids_ = ftids;

}
} // namespace mono