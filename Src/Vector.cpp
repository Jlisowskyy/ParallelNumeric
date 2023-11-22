// Author: Jakub Lisowski

#include "../Include/Types/Vector.hpp"

//-----------------------------------------
// AVX spec
//-----------------------------------------

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

#endif // __AVX__