// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMMainEngine.h"
#include "pcl/visualization/cloud_viewer.h"

using namespace ITMLib::Engine;

template<typename T>
T inline get_abs(T x){ return x < 0? -x:x; }


void ITMMainEngine::fetchCloud_test(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud,
							   ITMScene<ITMVoxel, ITMVoxelIndex> *_warped_scene) {

	int volume_x = _warped_scene->index.getVolumeSize().x;
	int volume_y = _warped_scene->index.getVolumeSize().y;
	int volume_z = _warped_scene->index.getVolumeSize().z;

//	const int DIVISOR = 32767;

#define FETCH(x, y, z) (_warped_scene->localVBA.GetVoxelBlocks()[(x) + (y) *volume_x + (z) * volume_x * volume_y])

	Eigen::Array3f cell_size(_warped_scene->sceneParams->voxelSize);

	Eigen::Vector3f translation_volumeCoo_to_liveFrameCoo(-volume_x*cell_size[0]/2, -volume_y*cell_size[1]/2, 0);


/*openMP shoule be opened*/
//#ifdef WITH_OPENMP
//#pragma omp parallel for
//#endif
	for (int x = 1; x < volume_x-1; x++){
		for (int y = 1; y < volume_y-1; y++){
			for (int z = 0; z < volume_z-1; z++){
				ITMVoxel voxel_tmp = FETCH(x, y, z);
				float F = ITMVoxel::SDF_valueToFloat(voxel_tmp.sdf); //[0,32767]
				int W = voxel_tmp.w_depth;//{0,1}  after integraing the live frame, W of allocated voxels should not be zero anymore

				if (W == 0 || F == 1) continue;

				Eigen::Vector3f V = ((Eigen::Array3i(x,y,z).cast<float>() + Eigen::Array3f(0.5f))*cell_size).matrix();

				int dz = 1;
				//for (int dy = -1; dy < 2; dy++){
					//for (int dx = -1; dx < 2; dx++){
                for (int dy = 0; dy < 1; dy++){
                    for (int dx = 0; dx < 1; dx++){
						ITMVoxel voxel = FETCH(x+dx, y+dy, z+dz);
						float Fn = ITMVoxel::SDF_valueToFloat(voxel.sdf); //[0,32767]
						int Wn = voxel.w_depth;

						//if (Wn == 0 || Fn == 1) continue;
//                        F * Fn == 0 || (F > 0 && Fn < 0)
						if (F * Fn == 0 || (F > 0 && Fn < 0)){
							Eigen::Vector3f Vn = ((Eigen::Array3i (x+dx, y+dy, z+dz).cast<float>() + Eigen::Array3f(0.5f)) * cell_size).matrix();
							Eigen::Vector3f point;
							if (F == 0 && Fn ==0){//in volume coo
#if 0
                                {
                                    int cnt = 0; int F[8];
                                    ITMVoxel voxel1 = FETCH(x+2, y, z);
                                    F[0] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y, z);
                                    F[1] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x, y+2, z);
                                    F[2] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x, y-2, z);
                                    F[3] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x+2, y+2, z);
                                    F[4] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y-2, z);
                                    F[5] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y+2, z);
                                    F[6] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x+2, y-2, z);
                                    F[7] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    for(int i=0; i < 8; i++)
                                        if(F[i]==0) cnt++;
                                    if(cnt < 1) continue;
                                }
#endif
                                point = (V + Vn) / 2;
							}
							else{ //(F==0,Fn!=0) or (F!=0,Fn==0)
#if 0
                                //check nearest voxel to find noise
                                {
                                    int cnt = 0; int F[8];
                                    ITMVoxel voxel1 = FETCH(x+2, y, z);
                                    F[0] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y, z);
                                    F[1] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x, y+2, z);
                                    F[2] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x, y-2, z);
                                    F[3] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x+2, y+2, z);
                                    F[4] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y-2, z);
                                    F[5] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x-2, y+2, z);
                                    F[6] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    voxel1 = FETCH(x+2, y-2, z);
                                    F[7] = ITMVoxel::SDF_valueToFloat(voxel1.sdf); //[0,32767]
                                    for(int i=0; i < 8; i++)
                                        if(F[i]==0) cnt++;
                                    if(cnt < 1) continue;
//                              }
#endif
                                point = (V * float(get_abs(Fn)) + Vn * float(get_abs(F))) / float(get_abs(F) + get_abs(Fn));
							}


							point = (point + translation_volumeCoo_to_liveFrameCoo) * 1000; //mm

							pcl::PointXYZ xyz(point[0],point[1],point[2]);

							extracted_cloud->push_back(xyz);
						}
					}
				}
			}
		}
	}


}



//
//void ITMMainEngine::fetchCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud,
//							   ITMScene<ITMVoxel, ITMVoxelIndex> *_warped_scene) {
//
//    int volume_x = _warped_scene->index.getVolumeSize().x;
//	int volume_y = _warped_scene->index.getVolumeSize().y;
//	int volume_z = _warped_scene->index.getVolumeSize().z;
//
////	const int DIVISOR = 32767;
//
//#define FETCH(x, y, z) (_warped_scene->localVBA.GetVoxelBlocks()[(x) + (y) *volume_x + (z) * volume_x * volume_y])
//
//	Eigen::Array3f cell_size(_warped_scene->sceneParams->voxelSize);
//
//    Eigen::Vector3f translation_volumeCoo_to_liveFrameCoo(-volume_x*cell_size[0]/2, -volume_y*cell_size[1]/2, 0);
//
//
///*openMP shoule be opened*/
//#ifdef WITH_OPENMP
//#pragma omp parallel for
//#endif
//	for (int x = 1; x < volume_x-1; x++){
//		for (int y = 1; y < volume_y-1; y++){
//			for (int z = 0; z < volume_z-1; z++){
//				ITMVoxel voxel_tmp = FETCH(x, y, z);
//                float F = ITMVoxel::SDF_valueToFloat(voxel_tmp.sdf); //[0,32767]
//				int W = voxel_tmp.w_depth;//{0,1}  after integraing the live frame, W of allocated voxels should not be zero anymore
//
//				if (W == 0 || F == 1) continue;
//
//				Eigen::Vector3f V = ((Eigen::Array3i(x,y,z).cast<float>() + Eigen::Array3f(0.5f))*cell_size).matrix();
//
//				int dz = 1;
//				for (int dy = -1; dy < 2; dy++){
//					for (int dx = -1; dx < 2; dx++){
//						ITMVoxel voxel = FETCH(x+dx, y+dy, z+dz);
//                        float Fn = ITMVoxel::SDF_valueToFloat(voxel.sdf); //[0,32767]
//						int Wn = voxel.w_depth;
//
//                        if (Wn == 0 || Fn == 1) continue;
//
//                        if (F * Fn <= 0){
//                            Eigen::Vector3f Vn = ((Eigen::Array3i (x+dx, y+dy, z+dz).cast<float>() + Eigen::Array3f(0.5f)) * cell_size).matrix();
//                            Eigen::Vector3f point;
//                            if (F == 0 && Fn ==0){//in volume coo
//                                point = (V + Vn) / 2;
//                            }
//                            else{
//                                point = (V * (float)abs (Fn) + Vn * (float)abs (F)) / (float)(abs (F) + abs (Fn));
//                            }
//
//
//                            point = (point + translation_volumeCoo_to_liveFrameCoo) * 1000; //mm
//
//                            pcl::PointXYZ xyz(point[0],point[1],point[2]);
//
//                            extracted_cloud->push_back(xyz);
//                        }
//					}
//				}
//			}
//		}
//	}
//
//
//}



//imgSize_rgb:(640,480)
ITMMainEngine::ITMMainEngine(const ITMLibSettings *settings, const ITMRGBDCalib *calib, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string output_fname,
							 Vector2i imgSize_rgb, Vector2i imgSize_d):extracted_cloud(new pcl::PointCloud<pcl::PointXYZ>){
//	pcl::PointCloud<pcl::PointXYZ>::Ptr initial_pc(new pcl::PointCloud<pcl::PointXYZ>::Ptr);
	this->extracted_cloud->points.reserve(100000);
	this->cloud = cloud;
	this->output_file_name = output_fname;
	// create all the things required for marching cubes and mesh extraction
	// - uses additional memory (lots!)
	static const bool createMeshingEngine = true;

	if ((imgSize_d.x == -1) || (imgSize_d.y == -1)) imgSize_d = imgSize_rgb;

	this->settings = settings;

	this->scene = new ITMScene<ITMVoxel, ITMVoxelIndex>(&(settings->sceneParams), settings->useSwapping, 
		settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU);

    this->_warped_scene = new ITMScene<ITMVoxel, ITMVoxelIndex>(&(settings->sceneParams), false, MEMORYDEVICE_CPU);


	meshingEngine = NULL;
	switch (settings->deviceType)
	{
	case ITMLibSettings::DEVICE_CPU:
		lowLevelEngine = new ITMLowLevelEngine_CPU();
		viewBuilder = new ITMViewBuilder_CPU(calib);
		visualisationEngine = new ITMVisualisationEngine_CPU<ITMVoxel, ITMVoxelIndex>(scene);
		if (createMeshingEngine) meshingEngine = new ITMMeshingEngine_CPU<ITMVoxel, ITMVoxelIndex>();
		break;
	case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
		lowLevelEngine = new ITMLowLevelEngine_CUDA();
		viewBuilder = new ITMViewBuilder_CUDA(calib);
		visualisationEngine = new ITMVisualisationEngine_CUDA<ITMVoxel, ITMVoxelIndex>(scene);
		if (createMeshingEngine) meshingEngine = new ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>();
#endif
		break;
	case ITMLibSettings::DEVICE_METAL:
#ifdef COMPILE_WITH_METAL
		lowLevelEngine = new ITMLowLevelEngine_Metal();
		viewBuilder = new ITMViewBuilder_Metal(calib);
		visualisationEngine = new ITMVisualisationEngine_Metal<ITMVoxel, ITMVoxelIndex>(scene);
		if (createMeshingEngine) meshingEngine = new ITMMeshingEngine_CPU<ITMVoxel, ITMVoxelIndex>();
#endif
		break;
	}

	mesh = NULL;
	if (createMeshingEngine) mesh = new ITMMesh(settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU);

	Vector2i trackedImageSize = ITMTrackingController::GetTrackedImageSize(settings, imgSize_rgb, imgSize_d);

	renderState_live = visualisationEngine->CreateRenderState(trackedImageSize);
	renderState_freeview = NULL; //will be created by the visualisation engine

	denseMapper = new ITMDenseMapper<ITMVoxel, ITMVoxelIndex>(settings);
	denseMapper->ResetScene(scene);

    _warped_denseMapper = new ITMDenseMapper<ITMVoxel, ITMVoxelIndex>(settings);
    _warped_denseMapper->ResetScene(_warped_scene);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory<ITMVoxel, ITMVoxelIndex>::Instance().Make(trackedImageSize, settings, lowLevelEngine, imuCalibrator, scene);
	trackingController = new ITMTrackingController(tracker, visualisationEngine, lowLevelEngine, settings);

	trackingState = trackingController->BuildTrackingState(trackedImageSize);
	tracker->UpdateInitialPose(trackingState);

	view = NULL; // will be allocated by the view builder

	fusionActive = true;
	mainProcessingActive = true;
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
	if (renderState_freeview!=NULL) delete renderState_freeview;

	delete scene;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;

	if (meshingEngine != NULL) delete meshingEngine;

	if (mesh != NULL) delete mesh;
}

ITMMesh* ITMMainEngine::UpdateMesh(void)
{
	if (mesh != NULL) meshingEngine->MeshScene(mesh, scene);
	return mesh;
}

void ITMMainEngine::SaveSceneToMesh(const char *objFileName)
{
	if (mesh == NULL) return;
	meshingEngine->MeshScene(mesh, scene);
	mesh->WriteSTL(objFileName);
}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement)
{
	// prepare image and turn it into a depth image
	if (imuMeasurement==NULL) viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter,settings->modelSensorNoise);
	else viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, imuMeasurement);

	if (!mainProcessingActive) return;

	// tracking
	trackingController->Track(trackingState, view); //ICP,get transformation between current frame and fusioned model

    /* visualize pointCloud */
//	pcl::visualization::CloudViewer viewer("Cloud Viewer");
//	viewer.showCloud(cloud);
//	while(!viewer.wasStopped()){}

    // build a new tsdf volume for the warped pointcloud, and fusion the live frame into _warped_scene
    if (fusionActive) _warped_denseMapper->_warped_ProcessFrame(view, trackingState, scene, renderState_live, cloud, _warped_scene);

    //using fetchCloud algorithm to extract pointcloud from _warped_scene
	fetchCloud_test(this->extracted_cloud, _warped_scene);

    /* visualize pointCloud */
//	pcl::visualization::CloudViewer viewer("Cloud Viewer");
//	viewer.showCloud(extracted_cloud);
//	while(!viewer.wasStopped()){}

    /*output extracted_cloud*/
	pcl::io::savePCDFileBinary(this->output_file_name, *extracted_cloud);
    exit(0);

	// fusions
	//if (fusionActive) denseMapper->ProcessFrame(view, trackingState, scene, renderState_live);

	// raycast to renderState_live for tracking and free visualisation
	//trackingController->Prepare(trackingState, view, renderState_live);


}



Vector2i ITMMainEngine::GetImageSize(void) const
{
	return renderState_live->raycastImage->noDims;
}

void ITMMainEngine::GetImage(ITMUChar4Image *out, GetImageType getImageType, ITMPose *pose, ITMIntrinsics *intrinsics)
{
	if (view == NULL) return;

	out->Clear();

	switch (getImageType)
	{
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) 
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
		out->ChangeDims(view->depth->noDims);
		if (settings->trackerType==ITMLib::Objects::ITMLibSettings::TRACKER_WICP)
		{
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depthUncertainty->UpdateHostFromDevice();
			ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex>::WeightToUchar4(out, view->depthUncertainty);
		}
		else
		{
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depth->UpdateHostFromDevice();
			ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex>::DepthToUchar4(out, view->depth);
		}

		break;
	case ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST:
	{
		ORUtils::Image<Vector4u> *srcImage = renderState_live->raycastImage;
		out->ChangeDims(srcImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);	
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	{
		IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME) type = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL) type = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
		if (renderState_freeview == NULL) renderState_freeview = visualisationEngine->CreateRenderState(out->noDims);

		visualisationEngine->FindVisibleBlocks(pose, intrinsics, renderState_freeview);
		visualisationEngine->CreateExpectedDepths(pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, type);

		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}

void ITMMainEngine::turnOnIntegration() { fusionActive = true; }
void ITMMainEngine::turnOffIntegration() { fusionActive = false; }
void ITMMainEngine::turnOnMainProcessing() { mainProcessingActive = true; }
void ITMMainEngine::turnOffMainProcessing() { mainProcessingActive = false; }
