//
// Created by Haoguang Huang on 18-4-25.
//

#include "node.h"
#include "math.h"

transformation::transformation(const transformation &T_i) {
    this->R = T_i.R; this->t = T_i.t;
}

transformation::transformation(const Eigen::Matrix3d &R_i, const Eigen::Vector3f &t_i) {
    this->R = R_i; this->t = t_i;
}

node::node(const node& node_i) {
    this->pos = node_i.pos;
    this->T_mat = node_i.T_mat;
    this->w = node_i.w;
}

node::~node() {}

node::node(const Eigen::Vector3f &pos_i, const float &w_i, const transformation &T_i) {
    this->pos = pos_i;
    this->w = w_i;
    this->T_mat.push_back(T_i);
}

node::node(const Eigen::Vector3f &pos_i, const float w_i, Eigen::Matrix3d R_i, Eigen::Vector3f t_i) {
    this->pos = pos_i;
    this->w = w_i;
    transformation tmp(R_i, t_i);
    this->T_mat.push_back(tmp);
}

float node::compute_control_extent(const Eigen::Vector3f &pts) {
    Eigen::Vector3f tmp = pts - this->pos;
    return std::exp(-tmp.norm()/2/this->w/this->w);
}
