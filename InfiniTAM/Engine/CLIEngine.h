// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../ITMLib/Engine/ITMMainEngine.h"
#include "../ITMLib/Utils/ITMLibSettings.h"
#include "../Utils/FileUtils.h"
#include "../Utils/NVTimer.h"

#include "ImageSourceEngine.h"
#include "IMUSourceEngine.h"


#include "pcl/point_types.h"
#include "pcl-1.8/pcl/io/pcd_io.h"

namespace InfiniTAM
{
	namespace Engine
	{
		class CLIEngine
		{
			static CLIEngine* instance;

			ITMLibSettings internalSettings;
			ImageSourceEngine *imageSource;
			IMUSourceEngine *imuSource;
			ITMMainEngine *mainEngine;

			StopWatchInterface *timer_instant;
			StopWatchInterface *timer_average;

		private:
			ITMUChar4Image *inputRGBImage; ITMShortImage *inputRawDepthImage;
			ITMIMUMeasurement *inputIMUMeasurement;

			int currentFrameNo;
		public:
			static CLIEngine* Instance(void) {
				if (instance == NULL) instance = new CLIEngine();
				return instance;
			}

			float processedTime;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;


			void Initialise(ImageSourceEngine *imageSource, IMUSourceEngine *imuSource, ITMMainEngine *mainEngine,
				ITMLibSettings::DeviceType deviceType, pcl::PointCloud<pcl::PointXYZ>::Ptr);

            void Initialise(ImageSourceEngine *imageSource, IMUSourceEngine *imuSource, ITMMainEngine *mainEngine,
                            ITMLibSettings::DeviceType deviceType);

			void Shutdown();

			void Run();
			bool ProcessFrame();
		};
	}
}
