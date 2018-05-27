//
// Created by Haoguang Huang on 18-5-14.
//

#ifndef INFINITAM_WARPFIELD_CUDA_H

/*data_device[locId] means the nearest node index of the locId_th voxel
 * "data_device[i] == -1" means the i_th voxel have no nearest node within specified radius.
 * */
class warpField_CUDA{
public:
    warpField_CUDA(const int nodenum, const int volume_size/*512*/, const float voxelsize/*0.005m*/, const float maxR2);

    ~warpField_CUDA();

    void reset();

    /* allocate the whole node position from host memory to device memory
     * -@nodePos_host:float[3*nodenum]. The position of i_th node is Vector3(nodePos_host[3i],nodePos_host[3i+1],nodePos_host[3i+2])
     * -@nodenum
     * */
    void setNodePosFromHost2Device(float* nodePos_host, const int& nodenum);

    ///update warpfield by computing distance between the locId_th voxel and node, find out the shortest distance and
    /// then save into data_device[locId].
    void updateWarpField();

    short intrivalNN(const int locId);


    const unsigned int warpField_total_size;//resolution of warpField is the same as canonical volume, etc. size*size*size. Normally 512*512*512
    const unsigned int warpField_size;//Normally 512
    const float voxelSize; //m
    const float maxNodeR2; //max node control radius

    short* data_device;
    short* data_host;

    float* nodePos_device;//1*(3*n)
    const int nodeNum;
};


#define INFINITAM_WARPFIELD_CUDA_H

#endif //INFINITAM_WARPFIELD_CUDA_H
