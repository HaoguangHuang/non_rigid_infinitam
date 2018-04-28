//
// Created by Haoguang Huang on 18-4-25.
//

#include "nodeGraph.h"
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl/visualization/cloud_viewer.h"
#include <numeric>
#include <iostream>
#include <sstream>
#include <string>

nodeGraph::nodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld) {
    node_mat.resize(Layers);

    for(int L = 0; L < Layers; L++){
        float radius = control_diameter[L];

        if(L == 0){
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cld, centroid);
            Eigen::Vector3f pc_center(centroid[0],centroid[1],centroid[2]);
            node _node(pc_center); _node.status = NODE_IN_TOP_LAYER; _node.father_id = -1;

            node_mat[0].push_back(_node);
        }
        else{
            pcl::PointCloud<pcl::PointXYZ>::Ptr cld_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(cld);
            sor.setLeafSize(radius, radius, radius);
            sor.filter(*cld_filtered);

            for(int i = 0; i < cld_filtered->size(); i++){
                Eigen::Vector3f pts(cld_filtered->points[i].x,cld_filtered->points[i].y,cld_filtered->points[i].z);
                node _node(pts);
                if(L == 0){ _node.status = NODE_IN_TOP_LAYER; _node.father_id = -1;}
                else {_node.status = NODE_NOT_IN_TOP_LAYER; _node.father_id = -1;}
                node_mat[L].push_back(_node);
            }
#if 0
            //visualization
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
            for(int i = 0; i < cld->size(); i++){
                pcl::PointXYZRGB pts;
                pts.x = cld->points[i].x; pts.y = cld->points[i].y; pts.z = cld->points[i].z;
                pts.r = 0; pts.g = 255; pts.b = 0;
                tmp->push_back(pts);
            }
            pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
            viewer.addPointCloud(tmp,"pointcloud1");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"pointcloud1");
            viewer.setBackgroundColor (0, 0, 0);
            for(int i = 0 ; i < cld_filtered->size(); i++){
                std::stringstream ss;
                ss<<"node_"<<i;
                pcl::PointXYZ pos(cld_filtered->points[i].x,cld_filtered->points[i].y,cld_filtered->points[i].z);
                std::cout<<ss.str()<<endl;
                viewer.addSphere(pos,5,255,0,0,ss.str());
            }
            viewer.spin();
#endif
        } /* if(L==0),else */

    }/* for(int L = 0; L < Layers; L++) */

}


nodeGraph::~nodeGraph() {
//    delete control_radius;
//    control_radius = NULL;

}

inline double computeDist(node node1, node node2){
    return (node1.pos - node2.pos).norm();
}

void nodeGraph::createNodeTree() {
    if(Layers < 2){
        std::cout<<"need not create node tree"<<std::endl;
        return;
    }

    for(int L_son = 1; L_son < Layers; L_son++){
        for(int short node_son = 0; node_son < node_mat[L_son].size(); node_son++){
            double dist = std::numeric_limits<double>::max();
            for(short node_father = 0; node_father < node_mat[L_son-1].size(); node_father++){
                //process node_father and node_son
                double curr_dist = computeDist(node_mat[L_son][node_son], node_mat[L_son-1][node_father]);
                if(curr_dist < dist){
                    dist = curr_dist;
                    node_mat[L_son][node_son].father_id = node_father;
                }
            }
        }
    }
}

