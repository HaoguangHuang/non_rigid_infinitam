//
// Created by Haoguang Huang on 18-4-26.
//

#include "ITMMainEngine.h"
#include "pcl/visualization/cloud_viewer.h"
//#include <iostream>

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

