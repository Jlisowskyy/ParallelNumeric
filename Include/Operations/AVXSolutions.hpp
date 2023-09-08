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

inline __m256d OmitCacheLoad(const double* Obj){
    return reinterpret_cast<__m256d>(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(Obj)));
}
#endif

#endif //PARALLELNUM_AVXSOLUTIONS_HPP
