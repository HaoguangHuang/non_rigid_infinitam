//
// Created by Haoguang Huang on 18-5-14.
//

#ifndef INFINITAM_SCENE_CUDA_H

#include "../../../Utils/ITMLibDefines.h"

class scene_CUDA{
public:
    scene_CUDA();

    ~scene_CUDA();

    void fetchCloud_CUDA();

    const float SDF_valueToFloat(const short sdf);//change [-1,1] to [-32767, 32767]

    const int volume_size;
    const int cell_size;
    const int offset[3];
    const int ITMVoxel_size;//byte

    ITMVoxel_s* dev_ptr;
    ITMVoxel_s* host_ptr;


};

//
//struct test_struct{
//    test_struct():sdf(5),weight(2){}
//
//    short sdf;
//    int weight;
//};
//
//class test_class{
//public:
//    test_class();
//
//    ~test_class();
//
//    void process();
//
//    void print();
//
//
//    const int N = 16;
//
//    test_struct* dev_ptr;
//    test_struct* host_ptr;
//};


#define INFINITAM_SCENE_CUDA_H
#endif //INFINITAM_SCENE_CUDA_H