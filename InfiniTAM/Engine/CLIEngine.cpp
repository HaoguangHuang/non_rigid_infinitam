// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "CLIEngine.h"

#include <string.h>

#include "../Utils/FileUtils.h"

using namespace InfiniTAM::Engine;
CLIEngine* CLIEngine::instance;

void CLIEngine::Initialise(ImageSourceEngine *imageSource, IMUSourceEngine *imuSource, ITMMainEngine *mainEngine,
	ITMLibSettings::DeviceType deviceType, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	this->imageSource = imageSource;
	this->imuSource = imuSource;
	this->mainEngine = mainEngine;

	this->currentFrameNo = 0;

	bool allocateGPU = false;
	if (deviceType == ITMLibSettings::DEVICE_CUDA) allocateGPU = true;

	inputRGBImage = new ITMUChar4Image(imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(imageSource->getDepthImageSize(), true, allocateGPU);
	inputIMUMeasurement = new ITMIMUMeasurement();

#ifndef COMPILE_WITHOUT_CUDA
	ITMSafeCall(cudaThreadSynchronize());
#endif

	sdkCreateTimer(&timer_instant);
	sdkCreateTimer(&timer_average);

	sdkResetTimer(&timer_average);

	printf("initialised.\n");
}

bool CLIEngine::ProcessFrame()
{
	if (!imageSource->hasMoreImages()) return false;
	imageSource->getImages(inputRGBImage, inputRawDepthImage);

	if (imuSource != NULL) {
		if (!imuSource->hasMoreMeasurements()) return false;
		else imuSource->getMeasurement(inputIMUMeasurement);
	}

	sdkResetTimer(&timer_instant);
	sdkStartTimer(&timer_instant); sdkStartTimer(&timer_average);

	//actual processing on the mailEngine
	if (imuSource != NULL) mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, this->cloud, inputIMUMeasurement);
	else mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, this->cloud);

#ifndef COMPILE_WITHOUT_CUDA
	ITMSafeCall(cudaThreadSynchronize());
#endif
	sdkStopTimer(&timer_instant); sdkStopTimer(&timer_average);

	float processedTime_inst = sdkGetTimerValue(&timer_instant);
	float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

	printf("frame %i: time %.2f, avg %.2f\n", currentFrameNo, processedTime_inst, processedTime_avg);

	currentFrameNo++;

	return true;
}

void CLIEngine::Run()
{
	while (true) {
		if (!ProcessFrame()) break;
	}
}

void CLIEngine::Shutdown()
{
	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	delete inputRGBImage;
	delete inputRawDepthImage;
	delete inputIMUMeasurement;

	delete instance;
}
