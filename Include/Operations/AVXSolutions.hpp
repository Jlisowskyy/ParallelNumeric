//
// Created by Jlisowskyy on 03/09/2023.
//

#ifndef PARALLELNUM_AVXSOLUTIONS_HPP
#define PARALLELNUM_AVXSOLUTIONS_HPP

#include <immintrin.h>

#ifdef __AVX__

inline double GetSubEl(__m256d& Obj, size_t Ind){
    return ((double*)&Obj)[Ind];
}

inline float GetSubEl(__m256& Obj, size_t Ind){
    return ((float*)&Obj)[Ind];
}

inline double HorSum(__m256d& Obj){
    return GetSubEl(Obj, 0) + GetSubEl(Obj, 1) + GetSubEl(Obj, 2) + GetSubEl(Obj, 3);
}

inline float HorSum(__m256& Obj){
    return GetSubEl(Obj, 0) + GetSubEl(Obj, 1) + GetSubEl(Obj, 2) + GetSubEl(Obj, 3) +
            GetSubEl(Obj, 4) + GetSubEl(Obj, 5) + GetSubEl(Obj,6 ) +GetSubEl(Obj, 7);
}
#endif

#endif //PARALLELNUM_AVXSOLUTIONS_HPP
