//
// Created by Haoguang Huang on 18-5-14.
//

#include "scene_CUDA.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <iostream>


//__global__ void process_kernel(test_struct* dev_ptr, const int N){
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    dev_ptr[tid].weight = tid;
//    dev_ptr[tid].sdf = tid * 2;
//}
//
//
//test_class::test_class() {
//    host_ptr = new test_struct[N];
//    cudaMalloc((void**)&dev_ptr, sizeof(test_struct) * N);
//}
//
//
//test_class::~test_class() {
//    delete this->host_ptr;
//    cudaFree(this->dev_ptr);
//}
//
//
/////__host__
//void test_class::process() {
//    cudaError_t cudaStatus;
//    cudaStatus = cudaMemcpy(dev_ptr, host_ptr, sizeof(test_struct)*N, cudaMemcpyHostToDevice);
//    if(cudaStatus != cudaSuccess){
//        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
//    }
//
//    //kernel
//    dim3 gridSize(1,1,1);
//    dim3 blockSize(16,1,1);
//    process_kernel<<<gridSize, blockSize>>>(dev_ptr, N);
//
//    cudaStatus = cudaMemcpy(host_ptr, dev_ptr, sizeof(test_struct)*N, cudaMemcpyDeviceToHost);
//    if(cudaStatus != cudaSuccess){
//        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
//    }
//
//    //output host_ptr
//    print();
//}
//
//
//void test_class::print() {
//    for(int i = 0; i < N; i++){
//        std::cout<<host_ptr[i].sdf<<"  "<<host_ptr[i].weight<<std::endl;
//    }
//}

