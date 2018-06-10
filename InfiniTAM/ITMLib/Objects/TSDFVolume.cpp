/*
 * This file is part of Kintinuous.
 *
 * Copyright (C) 2015 The National University of Ireland Maynooth and 
 * Massachusetts Institute of Technology
 *
 * The use of the code within this file and all code within files that 
 * make up the software that is Kintinuous is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email commercialisation@nuim.ie.
 *
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
 
#include "Volume.h"
#include <stdio.h>
#include <algorithm>
#include <Eigen/Core>
#include "TSDFVolume.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "ColorVolume.h"

using namespace ITMLib::Objects;

TsdfVolume::TsdfVolume(const Eigen::Vector3i resolution/*512*/, const float volume_size/*3m*/)
 : resolution_(resolution)
{
    voxelWrap = make_int3(0,0,0);

    int volume_x = resolution_(0);
    int volume_y = resolution_(1);
    int volume_z = resolution_(2);

    assert(volume_x == volume_y && volume_y == volume_z);

    volume_.create (volume_y * volume_z, volume_x);

//    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant (Volume::get().getVolumeSize()); //meters
    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant(volume_size); //meters

    const float    default_tranc_dist  = 0.02f; //meters

    setSize(default_volume_size);
    setTsdfTruncDist(default_tranc_dist);

    initVolume(volume_);

    reset();
}

void
TsdfVolume::setSize(const Eigen::Vector3f& size)
{  
    size_ = size;
    setTsdfTruncDist(tranc_dist_);
}

void
TsdfVolume::setTsdfTruncDist (float distance)
{
    float cx = size_(0) / resolution_(0);
    float cy = size_(1) / resolution_(1);
    float cz = size_(2) / resolution_(2);

    tranc_dist_ = std::max (distance, 2.1f * std::max (cx, std::max (cy, cz)));
}

DeviceArray2D<short>
TsdfVolume::data() const
{
    return volume_;
}

const Eigen::Vector3f&
TsdfVolume::getSize() const
{
    return size_;
}

const Eigen::Vector3i&
TsdfVolume::getResolution() const
{
    return resolution_;
}

const Eigen::Vector3f
TsdfVolume::getVoxelSize() const
{    
    return size_.array () / resolution_.array().cast<float>();
}

float
TsdfVolume::getTsdfTruncDist () const
{
    return tranc_dist_;
}

void
TsdfVolume::reset()
{
    initVolume(volume_);
}

DeviceArray<pcl::PointXYZRGB>
TsdfVolume::fetchCloud(DeviceArray<pcl::PointXYZRGB> & cloud_buffer,
                                 int3 & voxelWrap,
                                 PtrStep<uchar4> color_volume,
                                 int minX,
                                 int maxX,
                                 int minY,
                                 int maxY,
                                 int minZ,
                                 int maxZ,
                                 int3 realVoxelWrap,
                                 int subsample) const
{
    if(cloud_buffer.empty())
    {
        cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);
    }


    float3 device_volume_size = device_cast<const float3>(size_);

    size_t size = extractCloudSlice(volume_,
                                    device_volume_size,
                                    cloud_buffer,
                                    voxelWrap,
                                    color_volume,
                                    minX,
                                    maxX,
                                    minY,
                                    maxY,
                                    minZ,
                                    maxZ,
                                    subsample,
                                    realVoxelWrap);

    DeviceArray<pcl::PointXYZRGB> newBuffer(cloud_buffer.ptr(), size);

    return newBuffer;
}

void
TsdfVolume::downloadTsdf (std::vector<float>& tsdf) const
{
    tsdf.resize (volume_.cols() * volume_.rows());
    volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

    for(int i = 0; i < (int) tsdf.size(); ++i)
    {
        float tmp = reinterpret_cast<short2*>(&tsdf[i])->x;
        tsdf[i] = tmp/DIVISOR;
    }
}


void
TsdfVolume::downloadTsdf (std::vector<short>& tsdf) const
{
    tsdf.resize (volume_.cols() * volume_.rows());
//    volume_.download(&tsdf[0], volume_.cols() * sizeof(int));
    volume_.download(&tsdf[0], volume_.cols() * sizeof(short));
}


void
TsdfVolume::downloadTsdfAndWeighs (std::vector<float>& tsdf, std::vector<short>& weights) const
{
    int volumeSize = volume_.cols() * volume_.rows();
    tsdf.resize (volumeSize);
    weights.resize (volumeSize);
    volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

    for(int i = 0; i < (int) tsdf.size(); ++i)
    {
        short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
        tsdf[i] = (float)(elem.x)/DIVISOR;
        weights[i] = (short)(elem.y);
    }
}

void TsdfVolume::saveTsdfToDisk(std::string filename) const
{
    std::vector<float> tsdf;
    std::vector<short> weights;

    downloadTsdfAndWeighs(tsdf, weights);

    std::string tsdfName = filename;
    tsdfName += "_tsdf.bin";

    FILE * fp = fopen(tsdfName.c_str(), "wb+");

    fwrite(&tsdf[0], sizeof(float) * tsdf.size(), 1, fp);

    fclose(fp);

    std::string weightName = filename;
    weightName += "_weight.bin";

    fp = fopen(weightName.c_str(), "wb+");

    fwrite(&weights[0], sizeof(short) * weights.size(), 1, fp);

    fclose(fp);
}


/*********************************************************************/
/** \brief Copy transformation of each node from _nodeGraph into node_T_host
 * */
void get_Node_T_host(double* node_T_host, nodeGraph* _nodeGraph){
    const int N = _nodeGraph->node_mat.back().size();

    for(int n = 0; n < N; n++){
        Eigen::Matrix4d T = _nodeGraph->node_mat.back()[n].T_mat.back().cast();

        const int step = n * 12;
        for(int c = 0; c < 4; c++)
            for(int r = 0; r < 3; r++)
                node_T_host[step + c*3 + r] = T(r,c);
    }
}


void TsdfVolume::integrateCanonicalVolume(const ITMView * view,
                                          ITMScene<ITMVoxel, ITMVoxelIndex>* scene,
                                          nodeGraph* _nodeGraph,
                                          ColorVolume* _ColorVolume){
    Vector2i depthImgSize = view->depth->noDims;//(640,480)
    float voxelSize = scene->sceneParams->voxelSize;

    float mu_mm = scene->sceneParams->mu_mm;
//    int maxW = scene->sceneParams->maxW;

    float *depth = view->depth->GetData(MEMORYDEVICE_CPU);

    float* depth_devPtr;//depth map in GPU
    cudaMalloc((void**)&depth_devPtr, sizeof(float) * depthImgSize.x * depthImgSize.y);
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(depth_devPtr, depth, sizeof(float) * depthImgSize.x * depthImgSize.y, cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){
        fprintf(stderr, "cudaMemcpy failed!");
    }

    float4 proj_param; //[fx,fy,cx,cy]
    proj_param.x = view->calib->intrinsics_d.projectionParamsSimple.fx;
    proj_param.y = view->calib->intrinsics_d.projectionParamsSimple.fy;
    proj_param.z = view->calib->intrinsics_d.projectionParamsSimple.px;
    proj_param.w = view->calib->intrinsics_d.projectionParamsSimple.py;

    int3 offset; //[-256,-256,0]
    offset.x = scene->index.getIndexData()->offset.x;
    offset.y = scene->index.getIndexData()->offset.y;
    offset.z = scene->index.getIndexData()->offset.z;

    //get transformation of nodeGraph
    //node_T_host store [R,t] in column-major format. etc. [R00, R10, R20, R01, R11, R21, R02, R12, R22, t0, t1, t2]

    double* node_T_host = new double[12 * _nodeGraph->node_mat.back().size()];
    get_Node_T_host(node_T_host, _nodeGraph);

    double* node_T_dev;
    cudaMalloc((void**)&node_T_dev, sizeof(double) * 12 * _nodeGraph->node_mat.back().size());
    cudaStatus = cudaMemcpy(node_T_dev, node_T_host, sizeof(double) * 12 * _nodeGraph->node_mat.back().size(), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){
        fprintf(stderr, "cudaMemcpy failed!");
    }

    integrateCanonicalVolume_core(depth_devPtr,
                                  _nodeGraph->warpField_dev->data_device,
                                  this->data(),
                                  depthImgSize.x,
                                  depthImgSize.y,
                                  this->resolution_(0),
                                  proj_param,
                                  voxelSize,
                                  offset,
                                  node_T_dev,
                                  mu_mm,
                                  _ColorVolume->data());


    cudaDeviceSynchronize();

    cudaFree(depth_devPtr);

    delete node_T_host;
    cudaFree(node_T_dev);
}


void TsdfVolume::getExtractedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cld){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_buffer.download(cld_xyzrgb->points);//unit:meter
    cld_xyzrgb->resize(cloud_buffer.size());

    extracted_cld->clear();

    const int N = cld_xyzrgb->size();
    extracted_cld->resize(N);

#pragma unroll
    for(int i = 0; i < N; i++){
        extracted_cld->points[i].x = cld_xyzrgb->points[i].x * 1000.0f;
        extracted_cld->points[i].y = cld_xyzrgb->points[i].y * 1000.0f;
        extracted_cld->points[i].z = cld_xyzrgb->points[i].z * 1000.0f;
    }
}









