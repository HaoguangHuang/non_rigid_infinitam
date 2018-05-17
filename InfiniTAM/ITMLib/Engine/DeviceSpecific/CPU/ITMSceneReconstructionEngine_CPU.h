// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMSceneReconstructionEngine.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"


namespace ITMLib
{
	namespace Engine
	{
		template<class TVoxel, class TIndex>
		class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine < TVoxel, TIndex >
		{};

		template<class TVoxel>
		class ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine < TVoxel, ITMVoxelBlockHash >
		{
		protected:
			ORUtils::MemoryBlock<unsigned char> *entriesAllocType;
			ORUtils::MemoryBlock<Vector4s> *blockCoords;

		public:
			void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

			void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState, bool onlyUpdateVisibleList = false);

			void IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState);

			ITMSceneReconstructionEngine_CPU(void);
			~ITMSceneReconstructionEngine_CPU(void);
		};

		template<class TVoxel>
		class ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine < TVoxel, ITMPlainVoxelArray >
		{
		public:
			void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

			void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState, bool onlyUpdateVisibleList = false);

			void IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState);

			void _warped_IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
									const ITMRenderState *renderState, pcl::PointCloud<pcl::PointXYZ>::Ptr warped_cloud,
										   ITMScene<TVoxel, ITMPlainVoxelArray> *_warped_scene);

			void build_volume_for_warped_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr warped_cloud,
													ITMScene<TVoxel, ITMPlainVoxelArray> *_warped_scene,
													float voxelSize, unsigned int vol_resolution);

			ITMSceneReconstructionEngine_CPU(void);
			~ITMSceneReconstructionEngine_CPU(void);
		};
	}
}
