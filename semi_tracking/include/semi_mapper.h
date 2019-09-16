#ifndef SEMI_MAPPER_H_20181121_WEN
#define SEMI_MAPPER_H_20181121_WEN

#include <assert.h>
#include <algorithm>
#include "common.h"
#include "util.h"
#include "map.h"
#include "frame.h"
#include "semi_tracker.h"
using namespace std;
namespace mono
{
typedef Eigen::ArrayXXf ArrayF;
class SemiMapper
{
  public:
    //Constructor
    SemiMapper()
        {world = nullptr; mapping_interval=1; is_mapping = false;}
    SemiMapper(SemiMap *world):
                world(world), mapping_interval(1), is_mapping(true){}
    SemiMapper(SemiMap *world, int mapping_interval, bool is_mapping) :
              world(world), mapping_interval(mapping_interval), is_mapping(is_mapping) {}

    /* Make 3D Points with Multi Ray */
    void MultiViewTriangulate(const vector<Mat34*> *poses,
                              const vector<Mat3X> *pts,
                              Eigen::ArrayXXd *pts_4d);

    /* Mutliview Matching Points */
    void FindMatchedPointsMutliView(std::vector<Mat34*> *poses,
                                    std::vector<Mat3X> *pts,
                                    std::vector<uint>* matched_ids,
                                    std::vector<double>& edgeness);
    
    void ProcessMapping(uint m_interval);

    
  private:
    void SetMappingInterval(int mapping_interval){this-> mapping_interval = mapping_interval;};
    void SetMappingPossible(bool is_mapping){this-> is_mapping = is_mapping;};

    SemiMap *world;
    //mapping interval
    int mapping_interval;
    //if this value false , not generate 3d points
    bool is_mapping;
};

/* Make 3D Points with 2 diffrent rays */
void TriangulatePoints(const Mat34 &pose0, const Mat34 &pose1,
                       const Mat3X &pts0, const Mat3X &pts1,
                       Eigen::ArrayXXd *pts_4d);

/* find matching point using features ID for Trianguation */
void FindMatchedPoints(const Mat3X &pts1, const vector<uint> &ftids1,
                       const Mat3X &pts2, const vector<uint> &ftids2,
                       Mat3X *new_pts1, Mat3X *new_pts2,
                       vector<uint> *matched_ids);

/* conert to undistorted and normalized coordinate*/
void MakeNormalizedPoints(const std::vector<Edgel> &pts, Mat3X *normalized_pts,
                          const Mat3 &intrinsic,
                          const Vec2 &radial_distortion,
                          const Vec3 &tangential_distortion);

/* Make World Coordinate Points (Reconstruction.) */
double UpdateWorldPoints(MyFrame &prev_frm, MyFrame &curr_frm, SemiMap *world_ );
} // namespace mono
#endif