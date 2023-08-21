//
// Created by Jlisowskyy on 21/08/2023.
//

#include "../Include/Operations/MatrixMultiplication.hpp"

#define min(a, b) a > b ? b : a

template<>
void GPMM<double>::CCKernel8x6(const size_t HorizontalCord, const size_t VerticalCord, const size_t Dim2Off)
#define VectPartUpperPtr MatA + kk * MatASoL + VerticalCord
#define VectPartLowerPtr MatA + kk * MatASoL + VerticalCord + 4
#define LoadVectCoef(shift) MatB[HorizontalCord * MatBSoL + shift + kk]
#define OnTargetVectUpper(shift) *((__m256d*) (MatC + (HorizontalCord + shift) * MatCSoL + VerticalCord))
#define OnTargetVectLower(shift) *((__m256d*) (MatC + (HorizontalCord + shift) * MatCSoL + VerticalCord + 4))
{
    __m256d VectCoefBuff0;
    __m256d VectCoefBuff1;
    __m256d VectPartBuffUpper;
    __m256d VectPartBuffLower;

    __m256d ResVectBuffUpper0  = _mm256_setzero_pd();
    __m256d ResVectBuffUpper1  = _mm256_setzero_pd();
    __m256d ResVectBuffUpper2  = _mm256_setzero_pd();
    __m256d ResVectBuffUpper3  = _mm256_setzero_pd();
    __m256d ResVectBuffUpper4  = _mm256_setzero_pd();
    __m256d ResVectBuffUpper5  = _mm256_setzero_pd();
    __m256d ResVectBuffLower0  = _mm256_setzero_pd();
    __m256d ResVectBuffLower1  = _mm256_setzero_pd();
    __m256d ResVectBuffLower2  = _mm256_setzero_pd();
    __m256d ResVectBuffLower3  = _mm256_setzero_pd();
    __m256d ResVectBuffLower4  = _mm256_setzero_pd();
    __m256d ResVectBuffLower5  = _mm256_setzero_pd();

    const size_t LoopRange = min(Dim2, Dim2Off + Dim2Part);

    for(size_t kk = Dim2Off; kk < LoopRange; ++kk){
#ifndef __clang__
        __builtin_prefetch(VectPartUpperPtr + MatASoL);
        __builtin_prefetch(VectPartLowerPtr + MatASoL);
#endif

        VectPartBuffUpper = _mm256_load_pd(VectPartUpperPtr);
        VectPartBuffLower = _mm256_load_pd(VectPartLowerPtr);

        VectCoefBuff0 = _mm256_set1_pd(LoadVectCoef(0));
        VectCoefBuff1 = _mm256_set1_pd(LoadVectCoef(1));

        ResVectBuffUpper0 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff0, ResVectBuffUpper0);
        ResVectBuffLower0 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff0, ResVectBuffLower0);
        ResVectBuffUpper1 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff1, ResVectBuffUpper1);
        ResVectBuffLower1 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff1, ResVectBuffLower1);

        VectCoefBuff0 = _mm256_set1_pd(LoadVectCoef(2));
        VectCoefBuff1 = _mm256_set1_pd(LoadVectCoef(3));

        ResVectBuffUpper2 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff0, ResVectBuffUpper2);
        ResVectBuffLower2 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff0, ResVectBuffLower2);
        ResVectBuffUpper3 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff1, ResVectBuffUpper3);
        ResVectBuffLower3 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff1, ResVectBuffLower3);

        VectCoefBuff0 = _mm256_set1_pd(LoadVectCoef(4));
        VectCoefBuff1 = _mm256_set1_pd(LoadVectCoef(5));

        ResVectBuffUpper4 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff0, ResVectBuffUpper4);
        ResVectBuffLower4 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff0, ResVectBuffLower4);
        ResVectBuffUpper5 = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff1, ResVectBuffUpper5);
        ResVectBuffLower5 = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff1, ResVectBuffLower5);
    }

    OnTargetVectUpper(0) += ResVectBuffUpper0;
    OnTargetVectLower(0) += ResVectBuffLower0;
    OnTargetVectUpper(1) += ResVectBuffUpper1;
    OnTargetVectLower(1) += ResVectBuffLower1;
    OnTargetVectUpper(2) += ResVectBuffUpper2;
    OnTargetVectLower(2) += ResVectBuffLower2;
    OnTargetVectUpper(3) += ResVectBuffUpper3;
    OnTargetVectLower(3) += ResVectBuffLower3;
    OnTargetVectUpper(4) += ResVectBuffUpper4;
    OnTargetVectLower(4) += ResVectBuffLower4;
    OnTargetVectUpper(5) += ResVectBuffUpper5;
    OnTargetVectLower(5) += ResVectBuffLower5;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
template<>
void GPMM<double>::CCPerform()
    // Vertical and horizontal position refers to actually filling block on C matrix
{
    for (size_t VerticalOuter = 0; VerticalOuter < Dim1; VerticalOuter += Dim1Part){
        for(size_t Dim2Outer = 0; Dim2Outer < Dim2; Dim2Outer += Dim2Part){
            for(size_t HorizontalOuter = 0; HorizontalOuter < Dim3; HorizontalOuter += Dim3Part){
                for(size_t VerticalInner = 0; VerticalInner < Dim3; VerticalInner += 8){
                    for(size_t HorizontalInner = 0; HorizontalInner < Dim3Part; HorizontalInner += 6){
                        CCKernel8x6(HorizontalInner + HorizontalOuter, VerticalInner + VerticalOuter, Dim2Outer);
                    }
                }
            }
        }
    }
}
#pragma clang diagnostic pop