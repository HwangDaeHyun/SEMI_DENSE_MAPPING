#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <fstream>
#include <chrono>
#include "viewer.h"
#include "frame.h"
#include "util.h"
#include "semi_mapper.h"
#include "optimizer.h"
#include "evaluator.h"
#define MAX_COLS 512
using namespace std;
using namespace mono;

DEFINE_int32(start,0, "index of start frame");
DEFINE_int32(end,2911, "index of end frame");
DEFINE_int32(recon_interval, 10, "reconstruction frame interval ");
DEFINE_string(calib, "458.654,457.296,367.215,248.375,-0.28340811,0.07395907,0.00019359,1.76187114e-05,0.0", 
					"calibration parameters fx, fy, cx ,cy , k1, k2 ,p1, p2 , p3");
void LoadVINSPose(std::map<int,Mat34>& poses, const char* file_path);
void LoadGT_EuRoC(std::map<int,Mat34> &poses ,const char* file_path);
bool LoadDepthMap(const string& file_path, cv::Mat &depthMap);
bool MotionCheck(const Mat34& prev_pose , const Mat34& curr_pose );
int main(int argc, char **argv)
{
	google::ParseCommandLineFlags(&argc, &argv, true);
  	google::InitGoogleLogging(argv[0]);

	//this class values are core of slam 

	//world map
	SemiMap world_;

	//tracking (Extract Edge Point , Tracking Edge Point)
	EdgeTracker semi_tracker;
	
	//mapping process (Triangulation, Make normalize points for mapping ,Find Same Id Point Match)
	SemiMapper mapper_(&world_); 

	//optimze process (bundle adjustment, loop closing, pose alignment, etc...)
	Optimizer optimizer_(&world_);
	
	//MapViewer
	Viewer viewer_(FLAGS_start,FLAGS_end, &world_);

	double fx, fy, cx, cy, k1, k2,p1,p2,p3;

	if (FLAGS_calib.empty()){
    	LOG(ERROR) << "calib is not given (fx,fy,cx,cy,k0,k1,k2,k3,k4)";
    	return EXIT_FAILURE;
  	}
  	CHECK_EQ(9, sscanf(&FLAGS_calib[0], "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                     					 &fx,&fy,&cx,&cy,&k1,&k2,&p1,&p2,&p3));

	Mat3 intrinsic;
	Vec2 radial_d;
	Vec3 tangential_d;	
	
	intrinsic << fx, 0, cx,
				0, fy, cy,
				0, 0, 1;
	radial_d << k1,k2;
	tangential_d << p1,p2,p3;

	string img_path = string(argv[1]);
	string gt_path = string(argv[2]);

	MCImageRGB8 image;
	MCImageGray8 image_gray;
	MCImageGray8::ArrayType image_array;
	cv::Mat depth_map;
	
	//load gt_pose and frame number
	std::map<int,Mat34> gpose_map;
	LoadGT_EuRoC(gpose_map, gt_path.c_str());
	//LoadVINSPose(gpose_map, gt_path.c_str());
	MyFrame prev_frame(FLAGS_start-1, gpose_map.find(FLAGS_start-1)->second , 
						cv::Mat(), Mat3X(0,0),std::vector<uint>(), std::vector<double>(),std::set<uint>());
	//world_.all_frames.push_back(prev_frame);
	vector<Edgel> prev_feats ,curr_feats;

	int r_interval = 0;
	
	for (int im_idx = FLAGS_start; im_idx <= FLAGS_end; im_idx+=2)
	{
		//load image
		char path[256];
		sprintf(path, img_path.c_str(), im_idx);
		if(!ReadImageRGB8(path,&image))
			break;

		RGB8ToGray8(image, &image_gray);
		image_array = image_gray.GetPlane();

		
		//==== 1. Semidense Tracking 
		if (!semi_tracker.isInitialized()){
			semi_tracker.Setup(image_array,intrinsic, prev_frame.gt_pose, gpose_map.find(im_idx)->second);
		}
		else{
			semi_tracker.Process(image_array, prev_frame.gt_pose, gpose_map.find(im_idx)->second);
		}
		
		curr_feats = semi_tracker.features();
	
		Mat3X curr_npts;
		vector<uint> curr_ids;
		vector<double> curr_edgenesses;
		MakeNormalizedPoints(curr_feats,&curr_npts, intrinsic, radial_d, tangential_d);
		for(uint ft_idx =0; ft_idx < curr_feats.size(); ft_idx++){
			curr_ids.push_back(curr_feats[ft_idx].id);
			curr_edgenesses.push_back(curr_feats[ft_idx].edgeness);
		}
		if(curr_feats.size() <10000){
			semi_tracker.SetRedect(true); 
		}
		MyFrame curr_frm(im_idx, gpose_map.find(im_idx)->second, depth_map, curr_npts, curr_ids, curr_edgenesses,std::set<uint>());
		world_.all_frames.push_back(curr_frm);
		
        //motion check
        
		if(semi_tracker.isInitialized()){

		}
		//==== 2. Reconstruction
		if(im_idx!= FLAGS_start&& (r_interval%FLAGS_recon_interval==0) && 
									MotionCheck(world_.all_frames[world_.all_frames.size()-1-r_interval].gt_pose,curr_frm.gt_pose)){
			double enroll_ratio =UpdateWorldPoints(world_.all_frames[world_.all_frames.size()-1-r_interval], curr_frm, &world_);
			//mapper_.ProcessMapping(r_interval);
			
			//gt_idx = im_idx;
			if(enroll_ratio > 0.05){
				r_interval =1;
				world_.key_frames.push_back(curr_frm);
				semi_tracker.SetRedect(true); 
			}
			
			//===== 3. Optimize
			// optimizer_.SemiDenseOptimize();
		}
		r_interval++;
		
		//===== 4. Visualization
		viewer_.Process(im_idx, image_array, curr_feats, world_.all_frames);

		prev_feats = curr_feats;
		prev_frame = curr_frm;
	}
	
	return EXIT_SUCCESS;
}
bool MotionCheck(const Mat34& prev_pose , const Mat34& curr_pose ){
    //check motion
    Mat34 motion = RelativeTransform(prev_pose, curr_pose);

    Vec3 motion_trans = motion.block(0,3,3,1);

    double motion_dist = motion_trans.norm();

    std::cout << motion_dist<<std::endl;

    return motion_dist > 0.1;
}
void LoadGT_EuRoC(std::map<int,Mat34>& poses, const char* file_path){
    FILE *fp;
	fp = fopen(file_path, "r");
	if (fp == NULL)
	{
		return ;
	}
	int frame_num;
	double tx, ty ,tz ,qx, qy ,qz ,qw ;
	char str[MAX_COLS];
	int cnt= 0;
	
    Mat3 er;
    Vec3 et;
    er << 0.0148655429818, -0.999880929698, 0.00414029679422, 
          0.999557249008, 0.0149672133247, 0.025715529948,
         -0.0257744366974, 0.00375618835797, 0.999660727178;

    et << -0.0216401454975, -0.064676986768 ,0.00981073058949;

    Mat34 ext;
    ext.block(0,0,3,3) = er;
    ext.block(0,3,3,1) = et;
    
	while (fgets(str, MAX_COLS, fp) != NULL)
	{
		sscanf(str, "%04d %lf %lf %lf %lf %lf %lf %lf ", &frame_num, &tx, &ty, &tz, &qw, &qx, &qy, &qz);
        
		Eigen::Quaterniond q(qw , qx, qy, qz);
		Mat3 R= q.normalized().toRotationMatrix();
		Vec3 t;
		t <<tx,ty,tz;
		Mat34 pose;
		pose.block(0,0,3,3) = R;
		pose.block(0,3,3,1) = t;
		//cam2world
		poses.insert(pair<int, Mat34>(frame_num,InverseTransform(MergedTransform(pose,ext))));	
			
	}
    
	Mat34 start_pose = poses.find(24)->second;
	for(uint i = 24; i<poses.size(); i++){
		Mat34 Rt = RelativeTransform(start_pose, poses[i]);
	
		poses[i]= Rt;
	}
	
	fclose(fp);
}

void LoadVINSPose(std::map<int,Mat34>& poses, const char* file_path){
    FILE *fp;
	fp = fopen(file_path, "r");
	if (fp == NULL)
	{
		return ;
	}
	int frame_num;
	double tx, ty ,tz ,qx, qy ,qz ,qw ;
	char str[MAX_COLS];
	int cnt= 0;
	Mat3 er;
    Vec3 et;
    er << 0.0148655429818, -0.999880929698, 0.00414029679422, 
          0.999557249008, 0.0149672133247, 0.025715529948,
         -0.0257744366974, 0.00375618835797, 0.999660727178;

    et << -0.0216401454975, -0.064676986768 ,0.00981073058949;

    Mat34 ext;
    ext.block(0,0,3,3) = er;
    ext.block(0,3,3,1) = et;
	while (fgets(str, MAX_COLS, fp) != NULL)
	{
		sscanf(str, "%04d %lf %lf %lf %lf %lf %lf %lf ", &frame_num, &tx, &ty, &tz, &qw, &qx, &qy, &qz);
        
		Eigen::Quaterniond q(qw , qx, qy, qz);
		Mat3 R= q.normalized().toRotationMatrix();
		Vec3 t;
		t <<tx,ty,tz;
		Mat34 pose;
		pose.block(0,0,3,3) = R;
		pose.block(0,3,3,1) = t;

		poses.insert(pair<int, Mat34>(frame_num,pose));	

	}
    
	Mat34 start_pose = poses.find(FLAGS_start)->second;
	for(uint i = 0; i<poses.size(); i++){
		Mat34 Rt = RelativeTransform(poses[i], poses[i-1]);
		poses[i]= Rt;
	}
	
	fclose(fp);
}

