//
// Created by Jlisowskyy on 13/08/2023.
//

#include "../Include/Operations/NumericalCore.hpp"
#include "../Include/Operations/AVXSolutions.hpp"

// -----------------------------------------
// AVX specialisations
// -----------------------------------------

// -----------------------------------------
// Matrix sum
// -----------------------------------------

#ifdef __AVX__

template<>
void MatrixSumHelperAlignedArrays(double *Target, const double *const Input1, const double *const Input2,
                                  const size_t Elements) {
    const auto VectInput1 = (const __m256d*)Input1;
    const auto VectInput2 = (const __m256d*)Input2;
    auto VectTarget = (__m256d*)Target;

    const unsigned long VectSize = Elements / AVXInfo::f64Cap;
    for (size_t i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_pd(VectInput1[i], VectInput2[i]);
    }

    for (size_t i = VectSize * AVXInfo::f64Cap; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

template<>
void MatrixSumHelperAlignedArrays(float *Target, const float *const Input1, const float *const Input2,
                                  const size_t Elements) {
    const auto VectInput1 = (const __m256*)Input1;
    const auto VectInput2 = (const __m256*)Input2;
    auto VectTarget = (__m256*)Target;

    const size_t VectSize = Elements / AVXInfo::f32Cap;
    for (size_t i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_ps(VectInput1[i], VectInput2[i]);
    }

    for (size_t i = VectSize * AVXInfo::f32Cap; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

#endif // __AVX__

// ------------------------------------------
// Dot product
// ------------------------------------------

#ifdef __AVX__

template<>
double DotProductMachine<double>::DotProductAligned(size_t Begin, size_t End)
    // Highly limited by memory bandwidth
{
    using AVXInfo::f64Cap;
    static constexpr size_t LoadersCount = 3;

    __m256d AccRegs[AVXAccumulators] {_mm256_setzero_pd() };
    __m256d LoadRegs[LoadersCount] {_mm256_setzero_pd() };

    auto ApplyFMA = [&](size_t AccInd, size_t LoadInd, size_t Iter) -> void
        // Executes single operation on Chosen accumulator and Loader registers with respect to passed Index in data
    {
        AccRegs[AccInd] = _mm256_fmadd_pd(
                LoadRegs[LoadInd],
                _mm256_load_pd(BPtr + Iter + AccInd * f64Cap),
                AccRegs[AccInd]
        );
    };

    auto ProcessLoadAndApply = [&](size_t FirstAccInd, size_t Iter)
        // Does single blocked load operation and fma operation also
    {
        LoadRegs[0] = _mm256_load_pd(APtr + Iter + FirstAccInd * f64Cap);
        LoadRegs[1] = _mm256_load_pd(APtr + Iter + (FirstAccInd + 1) * f64Cap);
        LoadRegs[2] = _mm256_load_pd(APtr + Iter + (FirstAccInd + 2) * f64Cap);

        ApplyFMA(FirstAccInd, 0, Iter);
        ApplyFMA(FirstAccInd + 1, 1, Iter);
        ApplyFMA(FirstAccInd + 2, 2, Iter);
    };


    for (size_t i = Begin; i < End; i+=GetKernelSize())
        // Accumulates values to 12 registers
    {
        ProcessLoadAndApply(0,i);
        ProcessLoadAndApply(3,i);
        ProcessLoadAndApply(6,i);
        ProcessLoadAndApply(9,i);
    }

    AccRegs[0] += AccRegs[1];
    AccRegs[2] += AccRegs[3];
    AccRegs[4] += AccRegs[5];
    AccRegs[6] += AccRegs[7];
    AccRegs[8] += AccRegs[9];
    AccRegs[10] += AccRegs[11];

    AccRegs[0] += AccRegs[2];
    AccRegs[4] += AccRegs[6];
    AccRegs[8] += AccRegs[10];

    AccRegs[0] += AccRegs[4];
    AccRegs[0] += AccRegs[8];

    return HorSum(AccRegs[0]);
}

// Czas 0.75 * 1024 * 1024 * 1024 wynosil przed 0.5, 10 prob
// 0.373
// ten sam przypadek 0.16 dla omp

#endif // __AVX__

// ------------------------------------------
// Outer product AVX Implementation
// ------------------------------------------

// ------------------------------------------
// Matrix x Vector Multiplication AVX Implementation
// ------------------------------------------

#if defined(__AVX__) && defined(__FMA__)

static constexpr size_t HorizontalPartition = 2048;

template<>
void VMM<double>::RMVKernel12x4(const size_t HorizontalCord, const size_t VerticalCord){
    __m256d ResVectBuff0 = _mm256_setzero_pd();
    __m256d ResVectBuff1 = _mm256_setzero_pd();
    __m256d ResVectBuff2 = _mm256_setzero_pd();
    __m256d ResVectBuff3 = _mm256_setzero_pd();
    __m256d ResVectBuff4 = _mm256_setzero_pd();
    __m256d ResVectBuff5 = _mm256_setzero_pd();
    __m256d ResVectBuff6 = _mm256_setzero_pd();
    __m256d ResVectBuff7 = _mm256_setzero_pd();

    const size_t Range = std::min(HorizontalCord + HorizontalPartition, MatACols);
    for(size_t i = HorizontalCord; i < Range; i += 4)
        #define LoadVectPart(offset) (__m256d)_mm256_stream_load_si256((__m256i*)(RefPtr + (VerticalCord + offset) * MatASoL))
    {
        const double* RefPtr = MatA + i;
        __m256d CoefBuff = _mm256_load_pd(VectB + i);

        ResVectBuff0 = _mm256_fmadd_pd(LoadVectPart(0), CoefBuff, ResVectBuff0);
        ResVectBuff1 = _mm256_fmadd_pd(LoadVectPart(1), CoefBuff, ResVectBuff1);
        ResVectBuff2 = _mm256_fmadd_pd(LoadVectPart(2), CoefBuff, ResVectBuff2);
        ResVectBuff3 = _mm256_fmadd_pd(LoadVectPart(3), CoefBuff, ResVectBuff3);
        ResVectBuff4 = _mm256_fmadd_pd(LoadVectPart(4), CoefBuff, ResVectBuff4);
        ResVectBuff5 = _mm256_fmadd_pd(LoadVectPart(5), CoefBuff, ResVectBuff5);
        ResVectBuff6 = _mm256_fmadd_pd(LoadVectPart(6), CoefBuff, ResVectBuff6);
        ResVectBuff7 = _mm256_fmadd_pd(LoadVectPart(7), CoefBuff, ResVectBuff7);
    }
#define ToDouble(avx_obj) ((double*)(&avx_obj))
#define ReturnDoubleAVXSum(avx_obj) ToDouble(avx_obj)[0] + ToDouble(avx_obj)[1] + ToDouble(avx_obj)[2] + ToDouble(avx_obj)[3]
    VectC[VerticalCord] += ReturnDoubleAVXSum(ResVectBuff0);
    VectC[VerticalCord + 1] += ReturnDoubleAVXSum(ResVectBuff1);
    VectC[VerticalCord + 2] += ReturnDoubleAVXSum(ResVectBuff2);
    VectC[VerticalCord + 3] += ReturnDoubleAVXSum(ResVectBuff3);
    VectC[VerticalCord + 4] += ReturnDoubleAVXSum(ResVectBuff4);
    VectC[VerticalCord + 5] += ReturnDoubleAVXSum(ResVectBuff5);
    VectC[VerticalCord + 6] += ReturnDoubleAVXSum(ResVectBuff6);
    VectC[VerticalCord + 7] += ReturnDoubleAVXSum(ResVectBuff7);
}

template<>
void VMM<double>::PerformRMV(){
    for(size_t i = 0; i < MatACols; i += HorizontalPartition){
        for(size_t j = 0; j < MatARows; j += 1048552){
            const size_t Range = std::min(MatARows, MatARows + 1048552);
#pragma omp parallel for
            for(size_t jj = j; jj < Range; jj += 8){
                RMVKernel12x4(i, jj);
            }
        }
    }
}

template<>
void VMM<double>::CMVKernel12x4(size_t HorizontalCord, size_t VerticalCord){

}

template<>
void VMM<double>::PerformCMV(){
//    for(size_t i = 0; i < MatACols; i+= 2048){
//        for(size_t j = 0; j < MatARows; j+=)
//    }
}

#endif