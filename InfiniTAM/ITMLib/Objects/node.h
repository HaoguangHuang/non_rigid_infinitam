//
// Created by Haoguang Huang on 18-4-25.
//

#ifndef INFINITAM_NODE_H

//#include "../Utils/ITMMath.h"
#include <Eigen/Core>
#include <vector>

using std::vector;

enum node_status{NODE_IN_TOP_LAYER, NODE_NOT_IN_TOP_LAYER};

struct transformation{
    transformation();

    transformation(const transformation& T_i);

    transformation(const Eigen::Matrix3d& R_i, const Eigen::Vector3d& t_i);

    transformation(Eigen::Matrix4d& T_i);

    Eigen::Matrix4d cast(); //project [R,t] into 4*4 Transformation

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

class node{
public:
    //node() = default;

    ~node();

    node(const node&);

    node(const Eigen::Vector3d& pos_i, const float w_i = 1, Eigen::Matrix3d R_i = Eigen::Matrix<double,3,3>::Identity(),
        Eigen::Vector3d t_i = Eigen::Vector3d::Zero());

    node(const Eigen::Vector3d& pos_i, const float& w_i, const transformation& T_i);

    /* compute the extent of point controlling of node */
    float compute_control_extent(const Eigen::Vector3d& pts);

    inline void update_node_pose(const transformation& T_i){
        T_mat.push_back(T_i);
    }

    inline void update_node_pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t){
        transformation tmp(R, t);
        update_node_pose(tmp);
    }

    inline void update_node_pose(Eigen::Matrix4d T_i){
        transformation tmp(T_i);
        update_node_pose(tmp);
    }

    Eigen::Vector3d pos;           //dg_v. Record the initial position of node in the canonical frame
    float w;                       //dg_w
    vector<transformation> T_mat;  //dg_se3. T_mat[n] represents transformation from global coo into the (n+1)_th frame coo
    transformation T_last_to_live;
    node_status status;            //record whether this node is in TOP layer or not
    short father_id;               /*for node in TOP layer--->father_id = -1;
                                     for node not in TOP layer--->father_id > 0;*/


};
#define INFINITAM_NODE_H

#endif //INFINITAM_NODE_H
