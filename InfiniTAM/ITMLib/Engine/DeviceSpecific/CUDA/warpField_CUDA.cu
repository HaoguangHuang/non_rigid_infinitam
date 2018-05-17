#include <iostream>
#include <device_launch_parameters.h>
#include "warpField_CUDA.h"
//#include "ITMCUDAUtils.h"
#include "cuda_runtime.h"

#include <limits>

__device__ float computeDist2(float a[3], float b[3]){
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
}


/* -------------------- device ------------------------ */
__global__ void updateWarpField_kernel(short* data_dev, const int warpField_total_size, const int warpField_size,
                                        float* nodePos, const int nodeNum, const float voxelSize, const float maxR2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < 0 || y < 0 || x > warpField_size -1 || y > warpField_size - 1) return;

#pragma unroll(4)
    for(int z = 0; z < warpField_size; z++){
        int locId = z*warpField_size*warpField_size + y*warpField_size + x;
        if(locId < 0 || locId > warpField_total_size-1) continue;

        float voxel_pos[3];
        voxel_pos[0] = (float)(x - 256) * voxelSize * 1000.0f;
        voxel_pos[1] = (float)(y - 256) * voxelSize * 1000.0f;
        voxel_pos[2] = (float)(z) * voxelSize * 1000.0f;

        float minDist2 = 100000000000.0f;
        short NN_index = -1;//Nearest node index. when this value is equal to -1, it means this voxel has no nearest node within maxR2;

        for(int n = 0; n < nodeNum; n++){
            float nodepos[3];
            nodepos[0] = nodePos[3*n];
            nodepos[1] = nodePos[3*n+1];
            nodepos[2] = nodePos[3*n+2];

            float tmp = computeDist2(voxel_pos, nodepos);
            if(tmp < minDist2 && tmp < maxR2){
                minDist2 = tmp;
                NN_index = n;
            }
            data_dev[locId] = NN_index;
        }
    }
}


__global__ void intrivalNN_kernel(const int locId, short* NNid_dev, short* data_device){
    if(data_device == NULL){
        NNid_dev[0] = -1;
        return;
    }

    NNid_dev[0] = data_device[locId];
    return;
}


/* -------------------- host ------------------------ */
warpField_CUDA::warpField_CUDA(const int nodenum, const int volume_size, const float voxelsize, const float maxR2):
        warpField_total_size(volume_size*volume_size*volume_size),nodeNum(nodenum),warpField_size(volume_size), voxelSize(voxelsize),
        maxNodeR2(maxR2){

    this->data_host = new short[warpField_total_size];

    cudaMalloc((void**)&data_device, sizeof(short)*warpField_total_size);
    cudaMemset(data_device, -1, sizeof(short)*warpField_total_size);

    cudaMalloc((void**)&nodePos_device, sizeof(float)*nodeNum*3);
    cudaMemset(nodePos_device, 0, sizeof(float)*nodeNum*3);
}


warpField_CUDA::~warpField_CUDA() {
    delete data_host; data_host = NULL;

    cudaFree(data_device);
    cudaFree(nodePos_device);
}


void warpField_CUDA::reset() {
    if(data_host != NULL)
        delete data_host;
    data_host = new short[warpField_total_size];

    if(data_device == NULL)
        cudaMalloc((void**)&data_device, sizeof(short)*warpField_total_size);
    cudaMemset(data_device, -1, sizeof(short)*warpField_total_size);

    if(nodePos_device == NULL)
        cudaMalloc((void**)&nodePos_device, sizeof(float)*nodeNum*3);
    cudaMemset(nodePos_device, 0, sizeof(float)*nodeNum*3);
}


void warpField_CUDA::setNodePosFromHost2Device(float *nodePos_host, const int &nodenum) {
    if(nodenum != this->nodeNum){
        std::cout<<"node number is not equal!"<<std::endl;
        return;
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(nodePos_device, nodePos_host, sizeof(float)*nodeNum*3, cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){
        fprintf(stderr, "cudaMemcpy failed!");
    }

    return;
}


void warpField_CUDA::updateWarpField() {
    dim3 blockSize(32, 16);
    dim3 gridSize(warpField_size/blockSize.x, warpField_size/blockSize.y, 1);
    updateWarpField_kernel<<<gridSize, blockSize>> >(data_device, warpField_total_size, warpField_size,
                                                        nodePos_device, nodeNum, voxelSize, maxNodeR2);

    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(data_host, data_device, sizeof(short)*warpField_total_size, cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){
        fprintf(stderr, "cudaMemcpy failed!");
    }
    return;
}


short warpField_CUDA::intrivalNN(const int locId) {
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    short* NNid_dev;
    cudaMalloc((void**)&NNid_dev, sizeof(short));

    short* NNid_host = new short();

    intrivalNN_kernel<<<gridSize, blockSize>> >(locId, NNid_dev, data_device);

    cudaMemcpy(NNid_host, NNid_dev, sizeof(short), cudaMemcpyDeviceToHost);

    int NNid = *NNid_host;

    delete NNid_host;
    NNid_host = NULL;

    return NNid;

}


