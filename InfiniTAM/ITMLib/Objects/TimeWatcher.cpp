//
// Created by Haoguang Huang on 18-5-24.
//

#include "TimeWatcher.h"
#include <iostream>


void TimeWatcher::showTimeCost() {
    printf("--------This is frame %d--------\n", TimeWatcher::count[3]-1); //every frame should perform raycast step
    printf("CREATE_NODETREE AVERAGE TIME COST: %f ms\n", this->getAverageTime_ms(CREATE_NODETREE));
    printf("HIERARCHICAL_ICP AVERAGE TIME COST: %f ms\n", this->getAverageTime_ms(HIERARCHICAL_ICP));
    printf("INTEGRATE_VOLUME AVERAGE TIME COST: %f ms\n", this->getAverageTime_ms(INTEGRATE_VOLUME));
    printf("RAYCAST AVERAGE TIME COST: %f ms\n", this->getAverageTime_ms(RAYCAST));
    printf("FETCH_CLOUD AVERAGE TIME COST: %f ms\n", this->getAverageTime_ms(FETCH_CLOUD));
//    cout<<this->getAverageTime_ms(CREATE_NODETREE)<<" "<<this->getAverageTime_ms(INTEGRATE_VOLUME);

}

int TimeWatcher::count[5] = {0,0,0,0,0};
double TimeWatcher::total_time[5] = {0,0,0,0,0};