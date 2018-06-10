//
// Created by Haoguang Huang on 18-5-24.
//

#ifndef INFINITAM_TIMEWATCHER_H

#include <sys/time.h>
#include <cstdlib>

enum funcName{CREATE_NODETREE, HIERARCHICAL_ICP, INTEGRATE_VOLUME, RAYCAST, FETCH_CLOUD};

class TimeWatcher{
public:
    TimeWatcher(){};

    ~TimeWatcher();

    inline const double computeTimeCost(const timeval& tv_s, const timeval& tv_e){
        const double t_start_ms = tv_s.tv_sec*1000 + tv_s.tv_usec/1000;
        const double t_end_ms = tv_e.tv_sec*1000 + tv_e.tv_usec/1000;

        if(t_end_ms < t_start_ms) return 0;

        return t_end_ms - t_start_ms;
    }

    ///start time counting
    void TIC(const funcName funcname_In){
        gettimeofday(&time_start, nullptr);
        status = funcname_In;
    };


    ///end time counting
    void TOC(const funcName funcname_In){
        if(status != funcname_In) std::exit(0);//TODO: Not friendly at all...

        gettimeofday(&time_end, nullptr);

        const double timeCost_ms = computeTimeCost(time_start, time_end);

        count[funcname_In]++;
        total_time[funcname_In] += timeCost_ms;
    };

    double getAverageTime_ms(const funcName funcname_In){
        if(count[funcname_In] == 0) return 0;
        else{
            return total_time[funcname_In]/count[funcname_In]; //ms
        }
    };

    void showTimeCost(const unsigned int&);


    funcName status;

    timeval time_start;
    timeval time_end;

    static int count[5];
    static double total_time[5]; //ms

};



#define INFINITAM_TIMEWATCHER_H

#endif //INFINITAM_TIMEWATCHER_H
