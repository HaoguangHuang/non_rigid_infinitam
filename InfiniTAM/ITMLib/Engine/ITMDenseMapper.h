// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"
#include "../Utils/ITMLibSettings.h"

#include "../Objects/ITMScene.h"
#include "../Objects/ITMTrackingState.h"
#include "../Objects/ITMRenderState.h"

#include "../Engine/ITMSceneReconstructionEngine.h"
#include "../Engine/ITMVisualisationEngine.h"
#include "../Engine/ITMSwappingEngine.h"

#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"

#include "../Objects/nodeGraph.h"

namespace ITMLib
{
	namespace Engine
	{
		/** \brief
		*/
		template<class TVoxel, class TIndex>
		class ITMDenseMapper
		{
		private:
			ITMSceneReconstructionEngine<TVoxel,TIndex> *sceneRecoEngine;
			ITMSwappingEngine<TVoxel,TIndex> *swappingEngine;

		public:
			void ResetScene(ITMScene<TVoxel,TIndex> *scene);

			/// Process a single frame
			void ProcessFrame(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState_live);


			/// Update the visible list (this can be called to update the visible list when fusion is turned off)
			void UpdateVisibleList(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState);

			/** \brief Constructor
			    Ommitting a separate image size for the depth images
			    will assume same resolution as for the RGB images.
			*/
			explicit ITMDenseMapper(const ITMLibSettings *settings);
			~ITMDenseMapper();


			/*******************************************************/
			//psdf
			void integrateCanonicalVolume(const ITMView *view, ITMScene<TVoxel,TIndex> *scene, nodeGraph* );

            /*
             * @voxel:used for SDF value and weight update in canonical volume
             * @voxel_in_model_coo:[x, y, z, 1]. voxel position in homogeneous format
             * @projParams_d:[fx, fy, cx, cy]
             * @mu:truncated bandwidth. 0.02m
             * @maxW:100
             * @depth:depthMap. pixel in the i_th row, j_th col can be retrieved by depth[i*col+j]. Unit:mm
             * @depthImgSize:[640, 480]
             * @_nodeGraph
             * */
			void psdf(TVoxel& voxel, const Eigen::Vector4d& voxel_in_model_coo,
					  const Eigen::Vector4d& projParams_d, float mu, int maxW,
					  float* depth, const Eigen::Vector2i& depthImgSize,
					  nodeGraph* _nodeGraph, const int& locId);

			void psdfCore(TVoxel &voxel, const Eigen::Vector4d &projParams_d, Eigen::Matrix4d& T, const Eigen::Vector4d &pt_model,
                          float mu, int maxW,
                          float *depth, const Eigen::Vector2i &depthImgSize);

            /*
             * @_nodeGraph
             * @cld_lastFrame:extracted cloud from volume, and have already transformed into the last frame camera coo
             * @cld_live:pointcloud back-projected from live depth map
             * */
            void hierarchicalICP(nodeGraph* _nodeGraph, pcl::PointCloud<pcl::PointXYZ>::Ptr cld_lastFrame,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr cld_live);

			static long count;

		};
		template<class TVoxel, class TIndex>
		long ITMDenseMapper<TVoxel, TIndex>::count = 0;

	}
}

