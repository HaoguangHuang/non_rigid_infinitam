// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CPU.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"
#include "Eigen/Core"
#include "pcl/common/impl/centroid.hpp"
#include "pcl/visualization/cloud_viewer.h"


using namespace ITMLib::Engine;


template<typename T>
T inline get_abs(T x){ return x < 0? -x:x; }


void fetchCloud_teapot(pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud,
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
				for (int dy = -1; dy < 2; dy++){
					for (int dx = -1; dx < 2; dx++){
						ITMVoxel voxel = FETCH(x+dx, y+dy, z+dz);
						float Fn = ITMVoxel::SDF_valueToFloat(voxel.sdf); //[0,32767]
						int Wn = voxel.w_depth;

						//if (Wn == 0 || Fn == 1) continue;
//                        F * Fn == 0 || (F > 0 && Fn < 0)
						if (F * Fn == 0 || (F > 0 && Fn < 0)){
							Eigen::Vector3f Vn = ((Eigen::Array3i (x+dx, y+dy, z+dz).cast<float>() + Eigen::Array3f(0.5f)) * cell_size).matrix();
							Eigen::Vector3f point;
							if (F == 0 && Fn ==0){//in volume coo
								point = (V + Vn) / 2;
							}
							else{
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



template<class TVoxel>
void ITMSceneReconstructionEngine_CPU
		<TVoxel, ITMPlainVoxelArray>::build_volume_for_warped_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr warped_cloud,
																		ITMScene<TVoxel, ITMPlainVoxelArray> *_warped_scene,
																		float voxelSize, unsigned int vol_resolution){
	//find center of warped_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr locId_pc(new pcl::PointCloud<pcl::PointXYZ>);

	//compute locId
	for(int i = 0; i < warped_cloud->size(); i++){
		Eigen::Vector3f warped_pt(warped_cloud->points[i].x, warped_cloud->points[i].y, warped_cloud->points[i].z);
		Eigen::Vector3f id1(warped_pt[0]/voxelSize/1000, warped_pt[1]/voxelSize/1000, warped_pt[2]/voxelSize/1000); //m
//		Eigen::Vector3f offset(256,256,0);
		Eigen::Vector3f offset(vol_resolution/2,vol_resolution/2,0);
		Eigen::Vector3f res = id1 + offset;
		pcl::PointXYZ locId(int(res[0]+0.5), int(res[1]+0.5), int(res[2]+0.5));
		locId_pc->push_back(locId);
	}

	//build volume from locId_pc
	for(int i = 0; i < locId_pc->size(); i++){
		unsigned short x = locId_pc->points[i].x;
		unsigned short y = locId_pc->points[i].y;
		unsigned short z = locId_pc->points[i].z;

//		if (x < 0 || x > 511 || y < 0 || y > 511 || z < 0 || z > 511) continue;
		if (x < 0 || x > vol_resolution-1 || y < 0 || y > vol_resolution-1 || z < 0 || z > vol_resolution-1) continue;

		//if corresponding voxel is not empty
		int locId = _warped_scene->index.getVolumeSize().x * _warped_scene->index.getVolumeSize().y * z +
				_warped_scene->index.getVolumeSize().x * y + x;
		if(locId < 0) continue;
		TVoxel *voxelArray = _warped_scene->localVBA.GetVoxelBlocks();
		TVoxel voxel = voxelArray[locId];
//		int a = TVoxel::SDF_valueToFloat(voxel.sdf);
		if(voxelArray[locId].sdf == 32767){  //this voxel is empty
            voxelArray[locId].sdf = 0; voxelArray[locId].w_depth = 1;
		}
	}


//	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
//	fetchCloud_teapot(tmp,_warped_scene);
//	pcl::visualization::CloudViewer viewer("teapot in intial volume");
//	viewer.showCloud(tmp);
//	while(!viewer.wasStopped()){}

    //check how many voxels has zero-value sdf
//    int cnt = 0;
//    for (int i = 0; i < 512*512*512;i++){
//        TVoxel voxel = _warped_scene->localVBA.GetVoxelBlocks()[i];
//        if (voxel.sdf == 0) cnt++;
//    }
//
//	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pc(new pcl::PointCloud<pcl::PointXYZ>);
//	//check effect of volume-building
//	for (int locId = 0; locId < 512*512*512; locId++){
//		TVoxel voxel = _warped_scene->localVBA.GetVoxelBlocks()[locId];
//		if (voxel.sdf != 0) continue;
//		int z = locId / (_warped_scene->index.getVolumeSize().x*_warped_scene->index.getVolumeSize().y);
//		int tmp = locId - z * _warped_scene->index.getVolumeSize().x*_warped_scene->index.getVolumeSize().y;
//		int y = tmp / _warped_scene->index.getVolumeSize().x;
//		int x = tmp - y * _warped_scene->index.getVolumeSize().x;
//		Vector3f pt_model;
//
//		pt_model.x = float(x) * voxelSize;
//		pt_model.y = float(y) * voxelSize;
//		pt_model.z = float(z) * voxelSize;
//
//		pcl::PointXYZ p(pt_model[0],pt_model[1],pt_model[2]);
//
//		tmp_pc->push_back(p);
//	}

//	pcl::visualization::CloudViewer viewer("Cloud Viewer");
//	viewer.showCloud(tmp_pc);
//	while(!viewer.wasStopped()){}

};






template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CPU(void) 
{
	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	entriesAllocType = new ORUtils::MemoryBlock<unsigned char>(noTotalEntries, MEMORYDEVICE_CPU);
	blockCoords = new ORUtils::MemoryBlock<Vector4s>(noTotalEntries, MEMORYDEVICE_CPU);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CPU(void) 
{
	delete entriesAllocType;
	delete blockCoords;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

	ITMHashEntry tmpEntry;
	memset(&tmpEntry, 0, sizeof(ITMHashEntry));
	tmpEntry.ptr = -2;
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	for (int i = 0; i < scene->index.noTotalEntries; ++i) hashEntry_ptr[i] = tmpEntry;
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	for (int i = 0; i < SDF_EXCESS_LIST_SIZE; ++i) excessList_ptr[i] = i;

	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIds = renderState_vh->GetVisibleEntryIDs();
	int noVisibleEntries = renderState_vh->noVisibleEntries;

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int entryId = 0; entryId < noVisibleEntries; entryId++)
	{
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];

		if (currentHashEntry.ptr < 0) continue;

		globalPos.x = currentHashEntry.pos.x;
		globalPos.y = currentHashEntry.pos.y;
		globalPos.z = currentHashEntry.pos.z;
		globalPos *= SDF_BLOCK_SIZE;

		TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

		for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
		{
			Vector4f pt_model; int locId;

			locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == maxW) continue;
			//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) continue;

			pt_model.x = (float)(globalPos.x + x) * voxelSize;
			pt_model.y = (float)(globalPos.y + y) * voxelSize;
			pt_model.z = (float)(globalPos.z + z) * voxelSize;
			pt_model.w = 1.0f;

			ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(localVoxelBlock[locId], pt_model, M_d, 
				projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
		}
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	ITMHashEntry *hashTable = scene->index.GetEntries();
	ITMHashSwapState *swapStates = scene->useSwapping ? scene->globalCache->GetSwapStates(false) : 0;
	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();
	uchar *entriesAllocType = this->entriesAllocType->GetData(MEMORYDEVICE_CPU);
	Vector4s *blockCoords = this->blockCoords->GetData(MEMORYDEVICE_CPU);
	int noTotalEntries = scene->index.noTotalEntries;

	bool useSwapping = scene->useSwapping;

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	int lastFreeVoxelBlockId = scene->localVBA.lastFreeBlockId;
	int lastFreeExcessListId = scene->index.GetLastFreeExcessListId();

	int noVisibleEntries = 0;

	memset(entriesAllocType, 0, noTotalEntries);

	for (int i = 0; i < renderState_vh->noVisibleEntries; i++)
		entriesVisibleType[visibleEntryIDs[i]] = 3; // visible at previous frame and unstreamed

	//build hashVisibility
#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x*depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;
		buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
			invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable, scene->sceneParams->viewFrustum_min,
			scene->sceneParams->viewFrustum_max);
	}

	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
		//allocate
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx, exlIdx;
			unsigned char hashChangeType = entriesAllocType[targetIdx];

			switch (hashChangeType)
			{
			case 1: //needs allocation, fits in the ordered list
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;

				if (vbaIdx >= 0) //there is room in the voxel block array
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;

					hashTable[targetIdx] = hashEntry;
				}

				break;
			case 2: //needs allocation in the excess list
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;
				exlIdx = lastFreeExcessListId; lastFreeExcessListId--;

				if (vbaIdx >= 0 && exlIdx >= 0) //there is room in the voxel block array and excess list
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;

					int exlOffset = excessAllocationList[exlIdx];

					hashTable[targetIdx].offset = exlOffset + 1; //connect to child

					hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

					entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; //make child visible and in memory
				}

				break;
			}
		}
	}

	//build visible list
	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
	{
		unsigned char hashVisibleType = entriesVisibleType[targetIdx];
		const ITMHashEntry &hashEntry = hashTable[targetIdx];
		
		if (hashVisibleType == 3)
		{
			bool isVisibleEnlarged, isVisible;

			if (useSwapping)
			{
				checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
				if (!isVisibleEnlarged) hashVisibleType = 0;
			} else {
				checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
				if (!isVisible) { hashVisibleType = 0; }
			}
			entriesVisibleType[targetIdx] = hashVisibleType;
		}

		if (useSwapping)
		{
			if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
		}

		if (hashVisibleType > 0)
		{	
			visibleEntryIDs[noVisibleEntries] = targetIdx;
			noVisibleEntries++;
		}

#if 0
		// "active list", currently disabled
		if (hashVisibleType == 1)
		{
			activeEntryIDs[noActiveEntries] = targetIdx;
			noActiveEntries++;
		}
#endif
	}

	//reallocate deleted ones from previous swap operation
	if (useSwapping)
	{
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx;
			ITMHashEntry hashEntry = hashTable[targetIdx];

			if (entriesVisibleType[targetIdx] > 0 && hashEntry.ptr == -1) 
			{
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;
				if (vbaIdx >= 0) hashTable[targetIdx].ptr = voxelAllocationList[vbaIdx];
			}
		}
	}

	renderState_vh->noVisibleEntries = noVisibleEntries;

	scene->localVBA.lastFreeBlockId = lastFreeVoxelBlockId;
	scene->index.SetLastFreeExcessListId(lastFreeExcessListId);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::~ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;    //相机内参
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks(); //return the data ptr to voxel of volume

	const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z; ++locId)
	{//locid:traverse the whole voxels in volume one by one
		int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
		int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
		int y = tmp / scene->index.getVolumeSize().x;
		int x = tmp - y * scene->index.getVolumeSize().x;
		Vector4f pt_model;

		if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
		//if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;

		pt_model.x = (float)(x + arrayInfo->offset.x) * voxelSize;
		pt_model.y = (float)(y + arrayInfo->offset.y) * voxelSize;
		pt_model.z = (float)(z + arrayInfo->offset.z) * voxelSize;
		pt_model.w = 1.0f;
        //per voxel
		ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(voxelArray[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, 
			depth, depthImgSize, rgb, rgbImgSize); //tsdf fusion
	}
}


template class ITMLib::Engine::ITMSceneReconstructionEngine_CPU<ITMVoxel, ITMVoxelIndex>;


template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::_warped_IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
																					  const ITMTrackingState *trackingState, const ITMRenderState *renderState,
																							  pcl::PointCloud<pcl::PointXYZ>::Ptr warped_cloud,
																							  ITMScene<TVoxel, ITMPlainVoxelArray> *warped_scene)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;
    float vol_resolution = scene->sceneParams->vol_resolution;

//	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

//	M_d = trackingState->pose_d->GetM();
//	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;
	Matrix4f M_d = Matrix4f(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
	Matrix4f M_rgb = Matrix4f(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;    //相机内参
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

	/************build a volume for warped cloud. *************/
	//new volume for warped cloud is stored in pointer *scene
	ITMScene<TVoxel, ITMPlainVoxelArray> scene_backup(scene->sceneParams, scene->useSwapping, MEMORYDEVICE_CPU);

	build_volume_for_warped_pointcloud(warped_cloud, warped_scene, voxelSize, vol_resolution);


	scene = warped_scene;

	TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks(); //return the data ptr to voxel of volume

	const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();


//#ifdef WITH_OPENMP
//#pragma omp parallel for
//#endif
//	for (int locId = 0; locId < scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z; ++locId)
//	{//locid:traverse the whole voxels in volume one by one
//		int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
//		int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
//		int y = tmp / scene->index.getVolumeSize().x;
//		int x = tmp - y * scene->index.getVolumeSize().x;
//		Vector4f pt_model;
//
//		if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
//		//if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;
//
//		pt_model.x = (float)(x + arrayInfo->offset.x) * voxelSize;
//		pt_model.y = (float)(y + arrayInfo->offset.y) * voxelSize;
//		pt_model.z = (float)(z + arrayInfo->offset.z) * voxelSize;
//		pt_model.w = 1.0f;
//		//per voxel
//		ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(voxelArray[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW,
//																			 depth, depthImgSize, rgb, rgbImgSize); //tsdf fusion
//	}
}

