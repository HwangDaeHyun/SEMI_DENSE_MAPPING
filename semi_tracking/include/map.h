#ifndef MAP_H_20181121_WED
#define MAP_H_20181121_WED

#include <set>
#include <map>
#include <vector>
#include "common.h"
#include "frame.h"

namespace mono
{
struct MapPoint
{
  public:
    uint id;
    Vec3 pt3d;    
    double edgeness;

    std::vector<uint> frame_nums;
    
    std::vector<Vec3> history;


    MapPoint():id(-1),pt3d({0,0,0}),edgeness(0.0),frame_nums(std::vector<uint>()),history(std::vector<Vec3>()){};

    MapPoint(int id, Vec3 pt3d,  double edgeness,int frame_num):id(id), pt3d(pt3d),edgeness(edgeness){
        frame_nums.push_back(frame_num);
        history.push_back(pt3d);
    };
};

class SemiMap
{
  public:
    //ids and 3d Points
    std::map<int, MapPoint> semi_map;

    //all frame vector
    std::vector<MyFrame> all_frames;
    std::vector<MyFrame> key_frames;
    
};

} // namespace mono

#endif