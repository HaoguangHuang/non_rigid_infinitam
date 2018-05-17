//
// Created by Haoguang Huang on 18-4-26.
//

#include "ITMMainEngine.h"
#include "pcl/visualization/cloud_viewer.h"
#include "nabo/nabo.h"
//#include <iostream>
#include "../Objects/nodeGraph.cpp"

using namespace ITMLib::Engine;

void ITMMainEngine::transformUVD2XYZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cld, const ITMView* view){
    float* img = view->depth->GetData(MEMORYDEVICE_CPU); //m
    Vector2i imgSize = view->depth->noDims; //x = 640, y = 480

    double fx = view->calib->intrinsics_d.projectionParamsSimple.fx; //504
    double fy = view->calib->intrinsics_d.projectionParamsSimple.fy; //504
    double cx = view->calib->intrinsics_d.projectionParamsSimple.px; //352
    double cy = view->calib->intrinsics_d.projectionParamsSimple.py; //272

    //cld->is_dense = true;
    for(int y = 0; y < imgSize.y; y++)
        for(int x = 0; x < imgSize.x; x++){
            if(img[y*imgSize.x + x] > 0){
                double pt_z = img[y*imgSize.x + x] * 1000.0;
                double pt_x = (x - cx)*pt_z / fx;
                double pt_y = (y - cy)*pt_z / fy;
                pcl::PointXYZ pt(pt_x, pt_y, pt_z);
                cld->push_back(pt);
            }
        }
#if 0
    /* visualize pointCloud */
//	pcl::visualization::CloudViewer viewer("Cloud Viewer");
//	viewer.showCloud(cld);
//	while(!viewer.wasStopped()){}

//    std::cout<<"fx = "<<fx<<'\n'
//             <<"fy = "<<fy<<'\n'
//             <<"cx = "<<cx<<'\n'
//             <<"cy = "<<cy<<'\n';
//    system("pause");
//    int cnt = 0;
//    for(int i = 0 ; i < 300000 && cnt<50; i++){
//        if(img[i] > 0) {std::cout<<img[i]<<"  ";cnt++;}
//    }
//
//    std::cout<<"------imgSize.x="<<imgSize.x<<"  imgSize.y="<<imgSize.y<<std::endl;
//    system("pause");
#endif
}

void ITMMainEngine::boundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr cld_in) {
    float x_min = -350, x_max = 200;
    float y_min = -250, y_max = 100;
    float z_min = 700, z_max = 900;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cld_out(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i = 0; i < cld_in->size(); i++){
        if(cld_in->points[i].x >= x_min && cld_in->points[i].x <= x_max &&
           cld_in->points[i].y >= y_min && cld_in->points[i].y <= y_max &&
           cld_in->points[i].z >= z_min && cld_in->points[i].z <= z_max){
            cld_out->push_back(cld_in->points[i]);
        }
    }

    cld_in->clear();
    cld_in->swap(*cld_out);
}



template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::integrateCanonicalVolume(const ITMView *view, ITMScene<TVoxel,TIndex> *scene, nodeGraph* _nodeGraph){
    //Vector2i rgbImgSize = view->rgb->noDims;
    Vector2i depthImgSize = view->depth->noDims;
    float voxelSize = scene->sceneParams->voxelSize;

    //Matrix4f M_d; //camera ego-motion
    Vector4f projParams_d;

    projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;

    float mu = scene->sceneParams->mu_mm; int maxW = scene->sceneParams->maxW;

    float *depth = view->depth->GetData(MEMORYDEVICE_CPU);

    TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks(); //return the data ptr to voxel of volume
    const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();

    bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;

    const int N = scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z;

#pragma omp parallel for
    for (int locId = 0; locId < N; ++locId)
    {//locid:traverse the whole voxels in volume one by one
        int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
        int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
        int y = tmp / scene->index.getVolumeSize().x;
        int x = tmp - y * scene->index.getVolumeSize().x;

        Eigen::Vector4f voxel_in_model_coo = Eigen::Vector4f::Ones();

        if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
        //if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;

        voxel_in_model_coo(0) = (float)(x + arrayInfo->offset.x) * voxelSize * 1000.0f;
        voxel_in_model_coo(1) = (float)(y + arrayInfo->offset.y) * voxelSize * 1000.0f;
        voxel_in_model_coo(2) = (float)(z + arrayInfo->offset.z) * voxelSize * 1000.0f;
        voxel_in_model_coo(3) = 1.0f;
        //per voxel

        Eigen::Vector2i depthImgSize_eigen(depthImgSize.x, depthImgSize.y);
        Eigen::Vector4f proj_param(view->calib->intrinsics_d.projectionParamsSimple.fx,
                                   view->calib->intrinsics_d.projectionParamsSimple.fy,
                                   view->calib->intrinsics_d.projectionParamsSimple.px,
                                   view->calib->intrinsics_d.projectionParamsSimple.py);

        psdf(voxelArray[locId], voxel_in_model_coo, proj_param, mu, maxW, depth, depthImgSize_eigen, _nodeGraph, locId);

    }
}


template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::psdf(TVoxel& voxel, const Eigen::Vector4f &voxel_in_model_coo,
                                         const Eigen::Vector4f &projParams_d, float mu, int maxW,
                                         float *depth, const Eigen::Vector2i &depthImgSize, nodeGraph* _nodeGraph, const int& locId){
    //find KNN nodeGraph of input voxel in nodeGraph
    Eigen::VectorXi nodeIndice(_nodeGraph->noKNN);
    Eigen::VectorXf dist2(_nodeGraph->noKNN);

    Eigen::Vector3f voxel_in_model(voxel_in_model_coo(0),voxel_in_model_coo(1),voxel_in_model_coo(2));

//    bool haveKNN = _nodeGraph->findKNN(voxel_in_model, nodeIndice, dist2); //Use libnabo when find knn
//    bool haveKNN = _nodeGraph->findKNN_naive(voxel_in_model, nodeIndice, dist2);//Without libnabo when find knn
    //TODO: Here findKNN of warpField_dev can be only used when noKNN == 1.
    bool haveKNN = _nodeGraph->findKNN_CUDA(locId, nodeIndice, dist2); //findKNN

    ///if find, get transformation T of input voxel under the influence of warp field
    if(haveKNN){
        Eigen::Matrix4d T = _nodeGraph->warp(voxel_in_model, nodeIndice, dist2);

        //tranform the input voxel from global coo into live camera coo by T
        //psdf update
        psdfCore(voxel, projParams_d, T, voxel_in_model_coo, mu, maxW, depth, depthImgSize);
    }
    else{
        //if not find any KNN node of specified voxel within certain control radius, return.(Do nothing in this voxel)
        return;
    }

}


template<typename TVoxel, typename TIndex>
void ITMDenseMapper<TVoxel,TIndex>::psdfCore(TVoxel &voxel, const Eigen::Vector4f &projParams_d, Eigen::Matrix4d& T,
                                             const Eigen::Vector4f &voxel_in_model_coo, float mu, int maxW,
                                             float *depth, const Eigen::Vector2i &depthImgSize) {
    Eigen::Vector4f voxel_camera;
    Eigen::Vector2f voxel_image;
    float oldF, newF;
    int oldW, newW;

    //transform voxel from model coo into live camera coo
    Eigen::Vector4d voxel_in_live_camera_coo = T * voxel_in_model_coo.cast<double>();
    if(voxel_in_live_camera_coo(2) <= 0) return; //z<0 means this voxel is behind the camera view

    //project point into image
    float fx = projParams_d(0), fy = projParams_d(1), cx = projParams_d(2), cy = projParams_d(3);
    voxel_camera(0) = fx * voxel_in_live_camera_coo(0) / voxel_in_live_camera_coo(2) + cx; //u
    voxel_camera(1) = fy * voxel_in_live_camera_coo(1) / voxel_in_live_camera_coo(2) + cy; //v
    float &u = voxel_camera(0), &v = voxel_camera(1);
    if(u < 1 || v < 1 || u > depthImgSize(0)-1 || v > depthImgSize(1)-1) return; //depthImgSize = [640,480]

    //get depth measure from live depth map
    float depth_measure = depth[(int)(v+0.5)*depthImgSize(0) + (int)(u+0.5)] * 1000.0f; //check whether the unit is mm or not
    if(depth_measure <= 0) return;

    float diff = depth_measure - voxel_in_live_camera_coo(2);
    if(diff < -mu) return; //view frustum

    oldF = TVoxel::SDF_valueToFloat(voxel.sdf); oldW = voxel.w_depth;

    newF = MIN(1.0f, diff / mu);
    newW = 1;

    newF = oldW * oldF + newW * newF;
    newW = oldW + newW;
    newF /= newW;
    newW = MIN(maxW, newW);

    //write back
    voxel.sdf = TVoxel::SDF_floatToValue(newF);
    voxel.w_depth = newW;

}




template<typename T>
T inline get_abs(T x){ return x < 0? -x:x; }

void ITMMainEngine::fetchCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud,
                               ITMScene<ITMVoxel, ITMVoxelIndex> *scene) {
    if(!extracted_cloud->empty()){
        extracted_cloud->clear();
    }

    int volume_x = scene->index.getVolumeSize().x;
    int volume_y = scene->index.getVolumeSize().y;
    int volume_z = scene->index.getVolumeSize().z;

//	const int DIVISOR = 32767;

#define FETCH(x, y, z) (scene->localVBA.GetVoxelBlocks()[(x) + (y) *volume_x + (z) * volume_x * volume_y])

    Eigen::Array3f cell_size(scene->sceneParams->voxelSize);

    Eigen::Vector3f translation_volumeCoo_to_liveFrameCoo(-volume_x*cell_size[0]/2, -volume_y*cell_size[1]/2, 0);

/*openMP shoule be opened*/
//#ifdef WITH_OPENMP
//#pragma omp parallel for
//#endif
    for (int x = 1; x < volume_x-1; x++) {
        for (int y = 1; y < volume_y - 1; y++) {
            for (int z = 0; z < volume_z - 1; z++) {
                ITMVoxel voxel_tmp = FETCH(x, y, z);
                float F = ITMVoxel::SDF_valueToFloat(voxel_tmp.sdf); //[0,32767]
                int W = voxel_tmp.w_depth;//{0,1}  after integraing the live frame, W of allocated voxels should not be zero anymore

                Eigen::Vector3f V = ((Eigen::Array3i(x, y, z).cast<float>() + Eigen::Array3f(0.5f)) *
                                     cell_size).matrix();

                int dz = 1;
//                for (int dy = -1; dy < 2; dy++) {
//                    for (int dx = -1; dx < 2; dx++) {
                for (int dy = 0; dy < 1; dy++) {
                    for (int dx = 0; dx < 1; dx++) {
                        ITMVoxel voxel = FETCH(x + dx, y + dy, z + dz);
                        float Fn = ITMVoxel::SDF_valueToFloat(voxel.sdf); //[0,32767]
                        int Wn = voxel.w_depth;

//                        if (F * Fn <= 0) {
                        if (F > 0 && Fn < 0) {
                            Eigen::Vector3f Vn = (
                                    (Eigen::Array3i(x + dx, y + dy, z + dz).cast<float>() +
                                     Eigen::Array3f(0.5f)) *
                                    cell_size).matrix();
                            Eigen::Vector3f point;
                            if (F == 0 && Fn == 0) {//in volume coo
                                point = (V + Vn) / 2;
                            }
                            else { //(F==0,Fn!=0) or (F!=0,Fn==0)
                                point = (V * float(get_abs(Fn)) + Vn * float(get_abs(F))) /
                                        float(get_abs(F) + get_abs(Fn));
                            }

                            point = (point + translation_volumeCoo_to_liveFrameCoo) * 1000; //mm

                            pcl::PointXYZ xyz(point[0], point[1], point[2]);

                            extracted_cloud->push_back(xyz);
                        }
                    }
                }
            }
        }
    }
}



void ITMMainEngine::fetchCloud_parallel(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud,
                               ITMScene<ITMVoxel, ITMVoxelIndex> *scene) {
    if(!extracted_cloud->empty()){
        extracted_cloud->clear();
    }

    const int volume_x = scene->index.getVolumeSize().x;
    const int volume_y = scene->index.getVolumeSize().y;
    const int volume_z = scene->index.getVolumeSize().z;
    const int N = volume_x * volume_y * (volume_z - 1);
//	const int DIVISOR = 32767;

#define FETCH(x, y, z) (scene->localVBA.GetVoxelBlocks()[(x) + (y) *volume_x + (z) * volume_x * volume_y])

    Eigen::Array3f cell_size(scene->sceneParams->voxelSize);

    Eigen::Vector3f translation_volumeCoo_to_liveFrameCoo(-volume_x*cell_size[0]/2, -volume_y*cell_size[1]/2, 0);


#pragma omp parallel for
    for(int locId = 0; locId < N; locId++){
        int z = locId / (volume_x * volume_y);
        int tmp = locId - z * volume_x * volume_y;
        int y = tmp / volume_x;
        int x = tmp - y * volume_x;

        ITMVoxel voxel_tmp = FETCH(x, y, z);
        float F = ITMVoxel::SDF_valueToFloat(voxel_tmp.sdf); //[0,32767]
        int W = voxel_tmp.w_depth;//{0,1}  after integraing the live frame, W of allocated voxels should not be zero anymore

        Eigen::Vector3f V = ((Eigen::Array3i(x, y, z).cast<float>() + Eigen::Array3f(0.5f)) *
                             cell_size).matrix();

        int dz = 1, dy = 0, dx = 0;
        ITMVoxel voxel = FETCH(x + dx, y + dy, z + dz);
        float Fn = ITMVoxel::SDF_valueToFloat(voxel.sdf); //[0,32767]
        int Wn = voxel.w_depth;

        if (F > 0 && Fn < 0) {
            Eigen::Vector3f Vn = (
                    (Eigen::Array3i(x + dx, y + dy, z + dz).cast<float>() +
                     Eigen::Array3f(0.5f)) *
                    cell_size).matrix();
            Eigen::Vector3f point;
            if (F == 0 && Fn == 0) {//in volume coo
                point = (V + Vn) / 2;
            }
            else { //(F==0,Fn!=0) or (F!=0,Fn==0)
                point = (V * float(get_abs(Fn)) + Vn * float(get_abs(F))) /
                        float(get_abs(F) + get_abs(Fn));
            }

            point = (point + translation_volumeCoo_to_liveFrameCoo) * 1000; //mm

            pcl::PointXYZ xyz(point[0], point[1], point[2]);

            extracted_cloud->push_back(xyz);

        }
    }


}


