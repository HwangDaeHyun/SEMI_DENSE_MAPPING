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

DEFINE_int32(start,2, "index of start frame");
DEFINE_int32(end,200, "index of end frame");
DEFINE_int32(recon_interval,3, "reconstruction frame interval ");
DEFINE_string(calib, "329.115520046,329.11520046,320.0,240.0,0.0,0.0,0.0,0.0,0.0", 
					"calibration parameters fx, fy, cx ,cy , k1, k2 ,p1, p2 , p3");
DEFINE_double(motion_th,0.15, "index of end frame");
bool MotionCheck(const Mat34& prev_pose , const Mat34& curr_pose );
void LoadGTPose_ETH(std::map<int,Mat34> &poses ,const char* file_path);
bool LoadDepthMap(const string& file_path, cv::Mat &depthMap);
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
	string depth_path = string(argv[2]);
	string gt_path = string(argv[3]);

	MCImageRGB8 image;
	MCImageGray8 image_gray;
	MCImageGray8::ArrayType image_array;
	cv::Mat depth_map;
	
	char path[256];
	//load gt_pose and frame number
	std::map<int,Mat34> gpose_map;
	LoadGTPose_ETH(gpose_map, gt_path.c_str());
	sprintf(path, depth_path.c_str(), FLAGS_start-1);
		if(!LoadDepthMap(path, depth_map))
			return EXIT_FAILURE;
	MyFrame prev_frame(FLAGS_start-1, gpose_map.find(FLAGS_start-1)->second , depth_map, Mat3X(0,0),std::vector<uint>(), std::vector<double>(),std::set<uint>());
	//world_.all_frames.push_back(prev_frame);
	vector<Edgel> prev_feats ,curr_feats;

	int r_interval = 0;
	int gt_idx = 0;
	for (int im_idx = FLAGS_start; im_idx <= FLAGS_end; im_idx+=1)
	{
		//load image
		
		sprintf(path, img_path.c_str(), im_idx);
		if(!ReadImageRGB8(path,&image))
			break;

		RGB8ToGray8(image, &image_gray);
		image_array = image_gray.GetPlane();

		//load depth_map
		sprintf(path, depth_path.c_str(), im_idx);
		if(!LoadDepthMap(path, depth_map))
			break;

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

		MyFrame curr_frm(im_idx, gpose_map.find(im_idx)->second, depth_map, curr_npts, curr_ids, curr_edgenesses,std::set<uint>());

		ReprojectGTDepth(prev_frame, curr_frm ,intrinsic);
		world_.all_frames.push_back(curr_frm);
		if(semi_tracker.isInitialized()){
			ComputeCorresopndence(world_.all_frames[world_.all_frames.size()-1-r_interval],curr_frm,intrinsic);
		}
		//==== 2. Reconstruction
		if(im_idx!= FLAGS_start&& (r_interval%FLAGS_recon_interval==0) &&
			MotionCheck(world_.all_frames[world_.all_frames.size()-1-r_interval].gt_pose, curr_frm.gt_pose)
		){

			
			double enroll_ratio = UpdateWorldPoints(world_.all_frames[world_.all_frames.size()-1-r_interval],curr_frm, &world_);
			std::cout << enroll_ratio << std::endl;
			if(enroll_ratio > 0.05){
				r_interval =1;
				world_.key_frames.push_back(curr_frm);
				semi_tracker.SetRedect(false); 
			}
			//mapper_.ProcessMapping(r_interval);
			
			gt_idx = im_idx;
		}
		r_interval++;
		//optimizer_.SemiDenseOptimize();
	

		//===== 4. Visualization
		viewer_.Process(im_idx, image_array, curr_feats, world_.all_frames);

		prev_feats = curr_feats;
		prev_frame = curr_frm;
	}
	
	return EXIT_SUCCESS;
}

bool LoadDepthMap(const string& file_path, cv::Mat &depthMap)
{
	ifstream f(file_path.c_str());
	if(f.is_open()){

		double depth_arr[640 * 480];
		for (int i = 0; i < 640 * 480; i++)
		{
			f >> depth_arr[i];
		}
		cv::Mat tmp(480, 640, CV_64FC1, depth_arr);
		
		tmp.copyTo(depthMap);
		
		f.close();
	}else{
		std::cout << "depth file is not exist " << std::endl;
		return false;
	}
	return true;
}

void LoadGTPose_ETH(std::map<int,Mat34> &poses,const char *file_path)
{
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
	Mat34 start_pose;
	while (fgets(str, MAX_COLS, fp) != NULL)
	{
		sscanf(str, "%d %lf %lf %lf %lf %lf %lf %lf ", &frame_num, &tx, &ty, &tz, &qx, &qy, &qz, &qw);

		Eigen::Quaterniond q(qw , qx, qy, qz);
		Mat3 R= q.normalized().toRotationMatrix();
		Vec3 t;
		t <<tx,ty,tz;
		Mat34 pose;
		pose.block(0,0,3,3) = R;
		pose.block(0,3,3,1) = t;
		poses.insert(pair<int, Mat34>(frame_num, InverseTransform(pose)));	
		if(cnt++ == 0){
			start_pose = InverseTransform(pose);
		}
	}
	for(uint i = 1; i<poses.size(); i++){
		Mat34 Rt = RelativeTransform(start_pose, poses[i]);
		poses[i]= Rt;
	}
	
	fclose(fp);
}

bool MotionCheck(const Mat34& prev_pose , const Mat34& curr_pose ){
    //check motion
    Mat34 motion = RelativeTransform(prev_pose, curr_pose);

    Vec3 motion_trans = motion.block(0,3,3,1);

    double motion_dist = motion_trans.norm();

    std::cout << motion_dist<<std::endl;

    return motion_dist > 0.35;
}
