//
// Created by Haoguang Huang on 18-4-25.
//

#ifndef INFINITAM_NODEGRAPH_H

#include "node.h"
#include "../Engine/DeviceSpecific/CUDA/warpField_CUDA.h"

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"

#include "nabo/nabo.h"

#include <Eigen/Geometry>


class nodeGraph{
public:
    //create node tree
    explicit nodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld, const int volume_size, const float voxelsize);

    //nodeGraph() = default;
    ~nodeGraph();

    //find relationship between higher layer nodes and those in lower layer
    void createNodeTree();

    /*check current nodeGraph needed to be updated or not. If the number of points, that
     * don't belong to any nodes within control radius, is larger than OOR_thres, the nodeGraph
     * need to update.
     *
     * @extracted_cld:pointcloud that extracted from canonical volume
     * @cld_OOR:pointcloud that don't belong to any nodes
     * */
    bool checkUpdateOrNot(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cld,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr cld_OOR,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr cld_lastFrame);

    /*incrementally update nodeGraph
     *
     * @cld_OOR:pointcloud that don't belong to any nodes
     * */
    void updateNodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld_OOR);

    //create kdtree for nodes of the lowest layer
    void createNodeKDTree();

    //find knn of input point within maxRadiuFromNode2pts
    bool findKNN(const Eigen::Vector3f& voxel_in_model, Eigen::VectorXi& nodeIndice, Eigen::VectorXf& dist2);

    ///a version of findKNN without help of kdtree(libnabo)
    bool findKNN_naive(const Eigen::Vector3f& voxel_in_model, Eigen::VectorXi& nodeIndice, Eigen::VectorXf& dist2);

    ///a GPU implementation of findKNN_naive
    bool findKNN_CUDA(const int& locId, Eigen::VectorXi& nodeIndice, Eigen::VectorXf& dist2);

    //get Transformation of input point by warp field
    Eigen::Matrix4d warp(const Eigen::Vector3f& voxel_in_model, Eigen::VectorXi& nodeIndice, Eigen::VectorXf& dist2);

 /*
 * Paper:Markey-2017-Quaternion Averaging
 * Link: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
 * */
    Eigen::Quaterniond QuaternionInterpolation(Eigen::VectorXf& weight, Eigen::MatrixXd& quaternion_mat);

    ///Save node_mat[Layers-1] into array nodePos_host
    void update_NodePosHost();



    vector<vector<node> > node_mat;//the first vector represents different hierarchical layer
    float* nodePos_host;//1*(3*N)

    int currentFrameNo;
    static const unsigned short Layers = 4;
    const float control_diameter[Layers] = {2000,350,170,150};//mm {500,125,100,75}
    const unsigned short OOR_thres = 500;
    const float radius_cofficiency = 1.3;

    const int noKNN = 1;
    float maxRadiuFromNodeToPts2;//control_diameter[Layers-1]/2;

    Nabo::NNSearchF* node_kdtree;//nodes kdtree
    bool haveUpdateGraph;

//    Nabo::NNSearchF* voxel_tree;//3*(N*N*N)
//
//    Eigen::VectorXi warpfield; //1*(N*N*N). warpfield[locId] restore the nearest node indice of the locId-th voxel

    warpField_CUDA* warpField_dev;//1*(N*N*N). warpfield[locId] restore the nearest node indice of the locId-th voxel


};



#define INFINITAM_NODEGRAPH_H

#endif //INFINITAM_NODEGRAPH_H
