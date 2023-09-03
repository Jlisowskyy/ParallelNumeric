//
// Created by Jlisowskyy on 22/08/2023.
//

#include "../Include/Types/Vector.hpp"

//-----------------------------------------
// AVX spec
//-----------------------------------------

// TODO:: RECONSIDER modifying to decrease usage of dead threads

#ifdef __AVX__

template<>
Vector<double>& Vector<double>::sqrt(){
    ApplyAVXOnDataEffect<__m256d, &_mm256_sqrt_pd, &std::sqrt>();
    return *this;
}

template<>
Vector<float>& Vector<float>::sqrt(){
    ApplyAVXOnDataEffect<__m256, &_mm256_sqrt_ps, &sqrtf>();
    return *this;
}

template<>
Vector<float>& Vector<float>::reciprocal(){
    auto operand = [](float x) -> float{
        return 1 / x;
    };
    ApplyAVXOnDataEffect<__m256, &_mm256_rcp_ps, operand>();
    return *this;
}

template<>
Vector<float>& Vector<float>::reciprocal();

#endif // __AVX__