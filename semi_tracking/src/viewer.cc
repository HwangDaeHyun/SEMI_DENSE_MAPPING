#include "viewer.h"
#include <map>
DEFINE_bool(egocentric, false, "Visualize the model with the camera fixed.");
DEFINE_bool(show_all_keyframes, false, "Show all keyframes from the beginning.");
DEFINE_bool(show_reprojected_pts, false, "Show reprojected 3D points.");
DEFINE_double(display_cam_size, 0.3, "Display camera axis size.");
DEFINE_double(display_point_size, 0.8, "Display points size.");
DEFINE_string(out, "side2.csio", "Tracking result images.");

namespace mono
{
void Viewer::Process(int idx,
                     const MCImageGray8::ArrayType &image_array,
                     const vector<Edgel> &feats,
                     const vector<MyFrame> &frame_vec
                     )
{
    if (idx == this->start_idx && FLAGS_out.empty() == false)
    {
        vector<csio::ChannelInfo> channels;
        const int w = image_array.rows(), h = image_array.cols();
        channels.push_back(csio::ChannelInfo(
            0, csio::MakeImageTypeStr("rgb8", w, h), "output"));
        channels.push_back(csio::ChannelInfo(
            1, csio::MakeGeometricObjectTypeStr(w, h), "output"));
        map<string, string> config;
        if (this->csio_out.Setup(channels, config, FLAGS_out) == true)
        {
            
        }
        
    }

    if (this->csio_out.IsOpen())
    {
        
        MCImageRGB8 out;
        out.SetAllPlanes(image_array);

        DrawTextFormat(out, 5, 5, MakePixelRGB8(255, 255, 255), "%03d", idx);
        std::set<uint>::const_iterator citer;
    
        for (unsigned int i = 0; i < feats.size(); i++)
        {
            const Edgel &ft = feats[i];
            citer= frame_vec[frame_vec.size()-1].inlier_ids.find(ft.id);
            
            //draw epipole line
            
             Vec3 epl = ft.epl;
             double x1 = 1.0;
             double y1 = -(epl(0) * x1 + epl(2)) / epl(1);
             double x2 = out.width() - 1;
             double y2 = -(epl(0) * x2 + epl(2)) / epl(1);
             if(i%100 == 0)
             DrawLine(out, x1, y1, x2, y2, MCImageRGB8::MakePixel(255, 100, 100), 2);
             
             DrawDot(out, ft.pos(0), ft.pos(1), 
             MakePixelRGB8(frame_vec[frame_vec.size()-1].inlier_ids.end()==citer?100:250, 200 ,
                                      frame_vec[frame_vec.size()-1].inlier_ids.end()==citer? 255: 0), FLAGS_display_point_size );
        }
        
        int w = out.width();
        int h = out.height();

        DrawLine(out, 0, 0, w - 1, 0, MCImageRGB8::PixelType(255, 255, 255));
        DrawLine(out, 0, 0, 0, h - 1, MCImageRGB8::PixelType(255, 255, 255));
        DrawLine(out, w - 1, 0, w - 1, h - 1, MCImageRGB8::PixelType(255, 255, 255));
        DrawLine(out, 0, h - 1, w - 1, h - 1, MCImageRGB8::PixelType(255, 255, 255));

        vector<char> geom_buf;
        DrawGeometricOutput(&geom_buf, frame_vec);
        this->csio_out.PushSyncMark(2);
        this->csio_out.Push(0, out.data(), out.size() * 3);
        this->csio_out.Push(1, geom_buf.data(), geom_buf.size());
    }
}
template <class T>
inline string ToString(const T &v)
{
    stringstream ss;
    ss << v;
    return ss.str();
}

inline float *SetPointsToBuffer(const Mat &pts, float *buf)
{
    for (int j = 0; j < pts.cols(); ++j)
    {
        buf[0] = pts(0, j), buf[1] = pts(1, j), buf[2] = pts(2, j);
        buf += 3;
    }
    return buf;
}

inline float *SetPointsToBuffer(const Mat &pts, const int *col_idx, int num_pts,
                                float *buf)
{
    for (int j = 0; j < num_pts; ++j)
    {
        buf[0] = pts(0, col_idx[j]);
        buf[1] = pts(1, col_idx[j]);
        buf[2] = pts(2, col_idx[j]);
        buf += 3;
    }
    return buf;
}
void Viewer::DrawGeometricOutput(vector<char> *geom_buf_ptr, const vector<MyFrame>& frame_vec)
{
    vector<char> &geom_buf = *geom_buf_ptr;
    static map<int, Vec6> frm_poses;

    Vec6 pose = FLAGS_egocentric ? ToPoseVector(frame_vec[frame_vec.size()-1].gt_pose): (Vec6() << 0, 0, 0, 0, 0, 0).finished();
    if (FLAGS_egocentric)
    {
        Mat3 R = RotationRodrigues(pose.segment(0, 3));
        pose.segment(0, 3) << 0, 0, 0;
        pose.segment(3, 3) = R.transpose() * pose.segment(3, 3);
    }

    Mat34 pose0_mat = ToPoseMatrix(pose);
    
    
    const map<int, MapPoint> &semi_map = world->semi_map;
    
    if (FLAGS_show_all_keyframes == false)
        frm_poses.clear();

    for (uint i = 0; i <frame_vec.size(); ++i)
    {
        const MyFrame &frm =frame_vec[i];
        frm_poses[frm.frame_num] = ToPoseVector(frm.gt_pose);
    }
    const int num_kfs = frm_poses.size();
    //const int num_pts = ftid_pts_map.size();
    const int num_pts = semi_map.size();
    const int num_pred = 10;

    geom_buf.reserve(csio::ComputeGeometricObjectSize(1, num_pts,1) +
                     csio::ComputeGeometricObjectSize(1, num_kfs * 6, 1) +
                     csio::ComputeGeometricObjectSize(2, num_pred, 1) +
                     csio::ComputeGeometricObjectSize(1, 30, 1));

    vector<csio::GeometricObject::Opt> opts(1);
    opts[0] = csio::GeometricObject::MakeOpt(
        csio::GeometricObject::OPT_POINT_SIZE, 2);
    //Plot Semidense 3d Point

    // Plot 3D point cloud.
    csio::GeometricObject geom_pts =
            csio::AddGeometricObjectToBuffer('p', opts, num_pts, 1, &geom_buf);
    geom_pts.set_color(0, 0, 0, 0);
    float *pts_ptr = geom_pts.pts(0);
    int node_num = 0;
    for (map<int, MapPoint>::const_iterator it = semi_map.begin();
         it != semi_map.end(); ++it)
    {
        const Vec3 pt = pose0_mat * Hom(it->second.pt3d);
        //Set Different Color in Differenct Edgeness
        int edgeness = it->second.edgeness;
        pts_ptr = SetPointsToBuffer(pt, pts_ptr);
    }


    // Plot keyframe camera axes.
    csio::GeometricObject geom_cams =
        csio::AddGeometricObjectToBuffer('F', num_kfs * 16, 1, &geom_buf);
    geom_cams.set_color(0, 10, 10, 200, 20);
    float *cams_ptr = geom_cams.pts(0);
    // Make a camera 3D model.
    Eigen::Matrix<double, 4, 16> cam;
    double x = FLAGS_display_cam_size * 0.5;
    double y = x * 0.75;
    double z = FLAGS_display_cam_size * 0.3;
    cam.fill(0.0);
    cam(0, 1) = cam(0, 7) = cam(0, 8) = cam(0, 13) = cam(0, 14) = cam(0, 15) = x;
    cam(0, 3) = cam(0, 5) = cam(0, 9) = cam(0, 10) = cam(0, 11) = cam(0, 12) = -x;
    cam(1, 1) = cam(1, 3) = cam(1, 8) = cam(1, 9) = cam(1, 10) = cam(1, 15) = y;
    cam(1, 5) = cam(1, 7) = cam(1, 11) = cam(1, 12) = cam(1, 13) = cam(1, 14) = -y;
    cam.row(2).fill(z);
    cam(2, 0) = cam(2, 2) = cam(2, 4) = cam(2, 6) = 0;
    cam.row(3).fill(1.0);
    for (map<int, Vec6>::const_iterator it = frm_poses.begin();
         it != frm_poses.end(); ++it)
    {
        Mat34 pose_mat =ToPoseMatrix(it->second);
        Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
        cams_ptr = SetPointsToBuffer(pt, cams_ptr);
    }
    // Plot current camera pose.
    csio::GeometricObject geom_pose = csio::AddGeometricObjectToBuffer(
        'F', opts, 30, 1, &geom_buf);
    opts[0] = csio::GeometricObject::MakeOpt(
        csio::GeometricObject::OPT_LINE_WIDTH, 2);
    geom_pose.set_color(0, 255, 0, 0);
    
}
} // namespace mono
