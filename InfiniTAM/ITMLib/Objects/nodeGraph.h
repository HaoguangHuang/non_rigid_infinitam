//
// Created by Haoguang Huang on 18-4-25.
//

#ifndef INFINITAM_NODEGRAPH_H

#include "node.h"
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"



class nodeGraph{
public:
    //create node tree
    nodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld);

    //nodeGraph() = default;
    ~nodeGraph();

    //find relationship between higher layer nodes and those in lower layer
    void createNodeTree();

    //check current nodeGraph needed to be updated or not
    bool checkUpdateOrNot(pcl::PointCloud<pcl::PointXYZ>::Ptr cld);

    //incrementally update nodeGraph
    void update();

    //the first vector represents different hierarchical layer, the nested one represents those in
    vector<vector<node> > node_mat;
    int currentFrameNo;
    static const unsigned short Layers = 4;
    const float control_diameter[Layers] = {2000,350,170,150};//mm {500,125,100,75}
    const unsigned short OOR_thres = 500;
    const float radius_cofficiency = 1.3;

};

#define INFINITAM_NODEGRAPH_H

#endif //INFINITAM_NODEGRAPH_H
