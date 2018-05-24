//
// Created by Haoguang Huang on 18-4-25.
//

#include "nodeGraph.h"
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl/visualization/cloud_viewer.h"
#include <Eigen/Eigenvalues>


#include <numeric>
#include <iostream>
#include <sstream>
#include <string>

nodeGraph::nodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld, const int volume_size, const float voxelsize) {
    node_mat.resize(Layers);
    maxRadiuFromNodeToPts2 = control_diameter[Layers-1]/2 * control_diameter[Layers-1]/2;
    haveUpdateGraph = false;

    for(int L = 0; L < Layers; L++){
        if(L == 0){
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cld, centroid);
            Eigen::Vector3d pc_center(centroid[0],centroid[1],centroid[2]);
            node _node(pc_center); _node.status = NODE_IN_TOP_LAYER; _node.father_id = -1;

            node_mat[0].push_back(_node);
        }
        else{
            float leafsize = control_diameter[L];

            pcl::PointCloud<pcl::PointXYZ>::Ptr cld_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(cld);
            sor.setLeafSize(leafsize, leafsize, leafsize);
            sor.filter(*cld_filtered);

            for(int i = 0; i < cld_filtered->size(); i++){
                Eigen::Vector3d pts(cld_filtered->points[i].x,cld_filtered->points[i].y,cld_filtered->points[i].z);
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

    this->createNodeTree();
    this->createNodeKDTree();

    this->update_NodePosHost();

    std::cout<<"this->node_mat[Layers-1].size()="<<this->node_mat[Layers-1].size()<<std::endl;

    warpField_dev = new warpField_CUDA(this->node_mat[Layers-1].size(), volume_size, voxelsize, maxRadiuFromNodeToPts2);
    warpField_dev->setNodePosFromHost2Device(nodePos_host, node_mat[Layers-1].size());
    warpField_dev->updateWarpField();

}


nodeGraph::~nodeGraph() {
    delete warpField_dev;
    delete nodePos_host;

    warpField_dev = NULL;
    nodePos_host = NULL;
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
        for(int node_son = 0; node_son < node_mat[L_son].size(); node_son++){
            double dist = std::numeric_limits<double>::max();
            for(short node_father = 0; node_father < node_mat[L_son-1].size(); node_father++){
                //process node_father and node_son
                double curr_dist = computeDist(node_mat[L_son][node_son], node_mat[L_son-1][node_father]);
                if(curr_dist < dist){ //TODO:Here is not suitable to confirm that every node can own a father-node
                    dist = curr_dist;
                    node_mat[L_son][node_son].father_id = node_father;
                }
            }
        }
    }
}


void nodeGraph::createNodeKDTree() {
    const int nodeNum = node_mat[Layers-1].size();
    Eigen::MatrixXd M;
    M.resize(3,nodeNum);

    for(int i = 0; i < nodeNum; i++){
        M(0,i) = node_mat[Layers-1][i].pos(0);
        M(1,i) = node_mat[Layers-1][i].pos(1);
        M(2,i) = node_mat[Layers-1][i].pos(2);
    }

    node_kdtree = Nabo::NNSearchD::createKDTreeLinearHeap(M);
}


void nodeGraph::update_NodePosHost() {
    const int nodeNum = node_mat[Layers-1].size();

//    if(nodePos_host != NULL){
//        delete nodePos_host;
//    }
    nodePos_host = new float[3*nodeNum];

    for(int n = 0; n < nodeNum; n++){
        this->nodePos_host[3*n  ] = node_mat[Layers-1][n].pos(0);
        this->nodePos_host[3*n+1] = node_mat[Layers-1][n].pos(1);
        this->nodePos_host[3*n+2] = node_mat[Layers-1][n].pos(2);
    }
}


bool nodeGraph::findKNN(const Eigen::Vector3d& voxel_in_model, Eigen::VectorXi &nodeIndice, Eigen::VectorXd &dist2) {
    const int K = noKNN;
//    Eigen::VectorXd voxel_pos = voxel_in_model;
    node_kdtree->knn(voxel_in_model, nodeIndice, dist2, K);

    //exclude nodes that exceed control diameter from nodeIndice
    //TVoxel == ITMVoxel_s == ITMVoxel
    Eigen::VectorXi nodeIndice_filtered(K);
    Eigen::VectorXd dist2_filtered(K);

    int cnt = 0;
    for(int i = 0; i < dist2.size(); i++){
        if(dist2(i)<=maxRadiuFromNodeToPts2){
            nodeIndice_filtered(cnt) = nodeIndice(i);
            dist2_filtered(cnt) = dist2(i);
            cnt++;
        }
    }
    nodeIndice_filtered.conservativeResize(cnt);
    dist2_filtered.conservativeResize(cnt);

    nodeIndice = nodeIndice_filtered;
    dist2 = dist2_filtered;

    if(!cnt) return false;
    else return true;
}


//TODO: now only compute the 1-NN of specified node
bool nodeGraph::findKNN_naive(const Eigen::Vector3d& voxel_in_model, Eigen::VectorXi &nodeIndice, Eigen::VectorXd &dist2){
    const int K = noKNN;
    double min_dist = std::numeric_limits<double>::max();
    const int nodeNum = node_mat[Layers-1].size();

    for(int n = 0; n < nodeNum; n++){
        double diff = (node_mat[Layers-1][n].pos - voxel_in_model.cast<double>()).norm();
        if(diff < min_dist) {
            min_dist = diff;
            nodeIndice(0) = n;
            dist2(0) = min_dist * min_dist;
        }
    }

    if(dist2(0) < maxRadiuFromNodeToPts2){
        return true;
    }
    else return false;
}


//TODO: now only compute the 1-NN of specified node
bool nodeGraph::findKNN_CUDA(const int &locId, Eigen::VectorXi &nodeIndice, Eigen::VectorXd &dist2) {
//    const short NNid = warpField_dev->intrivalNN(locId);
    const short NNid = warpField_dev->data_host[locId];

    nodeIndice(0) = NNid;
    dist2(0) = 1; //TODO:...

    if(NNid == -1) return false;
    else return true;
}


Eigen::Matrix4d nodeGraph::warp(const Eigen::Vector3d& voxel_in_model, Eigen::VectorXi &nodeIndice, Eigen::VectorXd &dist2){
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const int N = nodeIndice.size(); //KNN number of this voxel

    if(N == 1){
        //when pts only have one nearest node, its transformation will be equal to that of nearest node.
        Eigen::Matrix3d R = this->node_mat[Layers-1][nodeIndice(0)].T_mat.back().R;
        Eigen::Vector3d t = this->node_mat[Layers-1][nodeIndice(0)].T_mat.back().t;
        T(0,0) = R(0,0); T(0,1) = R(0,1); T(0,2) = R(0,2); T(0,3) = t(0);
        T(1,0) = R(1,0); T(1,1) = R(1,1); T(1,2) = R(1,2); T(1,3) = t(1);
        T(2,0) = R(2,0); T(2,1) = R(2,1); T(2,2) = R(2,2); T(2,3) = t(2);
    }
    else{
        //compute normalized weight
        Eigen::VectorXd weight;
        weight.resize(N);
        for(int i = 0; i < N; i++){
            weight(i) = this->node_mat[Layers-1][nodeIndice(i)].compute_control_extent(voxel_in_model);
        }
        weight = weight / weight.sum();

        Eigen::Vector3d t_avg(0,0,0);
        Eigen::MatrixXd quaternion_mat;
        quaternion_mat.resize(N, 4);
        for(int i = 0; i < N; i++){
            Eigen::Matrix3d R = this->node_mat[Layers-1][nodeIndice(i)].T_mat.back().R;
            Eigen::Quaterniond q4(R);
            quaternion_mat(i,0) = q4.w(); quaternion_mat(i,1) = q4.x();
            quaternion_mat(i,2) = q4.y(); quaternion_mat(i,3) = q4.z();

            Eigen::Vector3d t = this->node_mat[Layers-1][nodeIndice(i)].T_mat.back().t;
            t_avg += t * weight(i);

        }

        Eigen::Quaterniond q_avg = QuaternionInterpolation(weight, quaternion_mat);

        Eigen::Matrix3d R_avg(q_avg); //or R_avg = q_avg.toRotationMatrix();

        T(0,0) = R_avg(0,0); T(0,1) = R_avg(0,1); T(0,2) = R_avg(0,2); T(0,3) = t_avg(0);
        T(1,0) = R_avg(1,0); T(1,1) = R_avg(1,1); T(1,2) = R_avg(1,2); T(1,3) = t_avg(1);
        T(2,0) = R_avg(2,0); T(2,1) = R_avg(2,1); T(2,2) = R_avg(2,2); T(2,3) = t_avg(2);
    }

    return T;

}


Eigen::Quaterniond nodeGraph::QuaternionInterpolation(Eigen::VectorXd& weight, Eigen::MatrixXd& quaternion_mat){
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    const int N = weight.size(); //KNN number of this voxel
    for(int i = 0; i < N; i++){
        Eigen::Vector4d q(quaternion_mat(i,0),
                          quaternion_mat(i,1),
                          quaternion_mat(i,2),
                          quaternion_mat(i,3));

        M = M + q * q.transpose() * weight(i);
    }

    Eigen::EigenSolver<Eigen::Matrix4d> es(M);
    Eigen::VectorXcd eigVec = es.eigenvectors().col(0);

    //FOR Quaterniond constructor func, its input parameters correspond to (w,x,y,z) in order.
    return Eigen::Quaterniond(eigVec(0).real(), eigVec(1).real(),eigVec(2).real(),eigVec(3).real());
}


inline pcl::PointXYZ getCloudLive(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const pcl::PointXYZ& cld_in){
    Eigen::Vector3d pts_in(cld_in.x, cld_in.y, cld_in.z);
    Eigen::Vector3d pts_out = R * pts_in + t;

    pcl::PointXYZ cld_out(pts_out(0),pts_out(1),pts_out(2));

    return cld_out;
}


///count the number of pointcloud that belonging to no nodes within specified radius
//use warpField to approximate
bool nodeGraph::checkUpdateOrNot(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cld,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr cld_OOR,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr cld_lastFrame) {
    if(!cld_OOR->empty())
        cld_OOR->clear();

    if(!cld_lastFrame->empty())
        cld_lastFrame->clear();

    int cnt = 0;
    const int volume_size = warpField_dev->warpField_size;
    const float voxelSize = warpField_dev->voxelSize;
    const float scale = voxelSize * 1000.0f;
    const int half_volume_size = volume_size / 2;

//TODO:pragma omp parallel for
    for(int i = 0; i < extracted_cld->size(); i++){
        const float x = extracted_cld->points[i].x; //mm
        const float y = extracted_cld->points[i].y; //mm
        const float z = extracted_cld->points[i].z; //mm

//        const int X = x / 1000.0f / voxelSize + volume_size / 2;
//        const int Y = y / 1000.0f / voxelSize + volume_size / 2;
//        const int Z = z / 1000.0f / voxelSize;
        const int X = x / scale + half_volume_size;
        const int Y = y / scale + half_volume_size;
        const int Z = z / scale;

        if(X < 0 || Y < 0 || Z < 0 || X > volume_size || Y > volume_size || Z > volume_size){
            std::cout<<"compute locId failed!"<< std::endl;
            continue;
        }

        //transform extracted cloud from model coo into last frame coo
        const int locId = Z * volume_size * volume_size + Y *volume_size + X;
        const int NNid = warpField_dev->data_host[locId];
        if(NNid == -1){
            cnt++;
            ///points without nearest node within specified control radius, should use transformation of the node from the lowest layer
            Eigen::Matrix3d R = this->node_mat[0][0].T_mat.back().R;
            Eigen::Vector3d t = this->node_mat[0][0].T_mat.back().t;
            cld_lastFrame->push_back(getCloudLive(R, t, extracted_cld->points[i]));
        }
        else{ //NNid >= 0
            Eigen::Matrix3d R = this->node_mat[Layers-1][NNid].T_mat.back().R;
            Eigen::Vector3d t = this->node_mat[Layers-1][NNid].T_mat.back().t;
            cld_lastFrame->push_back(getCloudLive(R, t, extracted_cld->points[i]));
        }
    }

    if(cnt > OOR_thres) return true;
    else return false;
}


void nodeGraph::updateNodeGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr cld_OOR) {
    ///perform downsample in cld_OOR, get the new nodes of the every layer
    for(int L = 1; L < Layers; L++){ //do nothing in the lowest layer
        float leafsize = control_diameter[L];

        pcl::PointCloud<pcl::PointXYZ>::Ptr cld_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cld_OOR);
        sor.setLeafSize(leafsize, leafsize, leafsize);
        sor.filter(*cld_filtered);

        for(int i = 0; i < cld_filtered->size(); i++){
            Eigen::Vector3d pts(cld_filtered->points[i].x,cld_filtered->points[i].y,cld_filtered->points[i].z);
            node _node(pts);
            _node.status = NODE_NOT_IN_TOP_LAYER; _node.father_id = -1;
            node_mat[L].push_back(_node);
        }
    }

    this->createNodeTree();
    this->createNodeKDTree();

    this->update_NodePosHost();
    warpField_dev->setNodePosFromHost2Device(nodePos_host, node_mat[Layers-1].size());
    warpField_dev->updateWarpField();
}