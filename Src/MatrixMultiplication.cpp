//
// Created by Jlisowskyy on 21/08/2023.
//

#include "../Include/Operations/MatrixMultiplication.hpp"

#define min(a, b) a > b ? b : a

template<>
inline void GPMM<double>::CCKernelXx6(const size_t HorizontalCord, const size_t VerticalCord, const size_t Dim2Off)
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

template<>
inline void GPMM<double>::CCKernelXxY(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off, size_t HorKernelSize){
    __m256d VectPartBuffUpper;
    __m256d VectPartBuffLower;
    __m256d VectCoefBuff;

    // Distance between Upper and Lower buffer of the corresponding vector
    static constexpr size_t TabOffset = (HorInBlockSize - 1);
    static constexpr size_t TabSize = 2 * TabOffset;
    __m256d ResultBuff[TabSize] = { _mm256_setzero_pd() };

    const size_t LoopRange = min(Dim2, Dim2Off + Dim2Part);

    for(size_t kk = Dim2Off; kk < LoopRange; ++kk){
#ifndef __clang__
        __builtin_prefetch(VectPartUpperPtr + MatASoL);
        __builtin_prefetch(VectPartLowerPtr + MatASoL);
#endif

        VectPartBuffUpper = _mm256_load_pd(VectPartUpperPtr);
        VectPartBuffLower = _mm256_load_pd(VectPartLowerPtr);

        for (size_t i = 0; i < HorKernelSize; ++i ){
            VectCoefBuff = _mm256_set1_pd(LoadVectCoef(i));

            ResultBuff[i] = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff, ResultBuff[i]);
            ResultBuff[i + TabOffset] = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff, ResultBuff[i + TabOffset]);
        }
    }

    for (size_t i = 0; i < HorKernelSize; ++i ){
        OnTargetVectUpper(i) += ResultBuff[i];
        OnTargetVectLower(i) += ResultBuff[i + TabOffset];
    }
}

template<>
inline void GPMM<double>::CCInnerParts(const size_t VerOut, const size_t HorOut, const size_t Dim2Outer){
    static constexpr size_t VerInBlockSize = 8;
    const size_t HorInMaxRange = min(HorOut + Dim3Part, Dim3);
    const size_t HorInFullyBlockedRange = (HorInMaxRange / HorInBlockSize ) * HorInBlockSize;
    const size_t VerInRange = min(VerOut + Dim1Part, Dim1);

    #pragma omp parallel for
    for(size_t VerIn = VerOut; VerIn < VerInRange; VerIn += VerInBlockSize){
        for(size_t HorIn = HorOut; HorIn < HorInFullyBlockedRange; HorIn += HorInBlockSize){
            CCKernelXx6(HorIn, VerIn, Dim2Outer);
        }
        if (HorInMaxRange != HorInFullyBlockedRange){
            CCKernelXxY(HorInFullyBlockedRange, VerIn, Dim2Outer, HorInMaxRange - HorInFullyBlockedRange);
        }
    }
}

template<>
inline void GPMM<double>::CCInnerPartsThreaded(size_t VerIn, size_t HorOut, size_t Dim2Outer){
    static constexpr size_t VerInBlockSize = 8;
    const size_t HorInMaxRange = min(HorOut + Dim3Part, Dim3);
    const size_t HorInFullyBlockedRange = (HorInMaxRange / HorInBlockSize ) * HorInBlockSize;

    for(size_t HorIn = HorOut; HorIn < HorInFullyBlockedRange; HorIn += HorInBlockSize){
       CCKernelXx6(HorIn, VerIn, Dim2Outer);
    }
    if (HorInMaxRange != HorInFullyBlockedRange){
        CCKernelXxY(HorInFullyBlockedRange, VerIn, Dim2Outer, HorInMaxRange - HorInFullyBlockedRange);
    }
}

template<>
void ThreadInstance<double>(GPMM<double>* Target, void (GPMM<double>::*Oper)(size_t, size_t, size_t)){
    Target->StartGuard->arrive_and_wait();

    while(!Target->WorkDone || !Target->CordQue.empty()){
        Target->QueGuard.lock();
        if (Target->CordQue.empty()){
            Target->QueGuard.unlock();
            continue;
        }

        P3D Cords = Target->CordQue.front();
        Target->CordQue.pop();
        Target->QueGuard.unlock();

        (Target->*Oper)(Cords.x, Cords.y, Cords.z);
    }
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
template<>
void GPMM<double>::CCPerform(unsigned ThreadCount)
    // Vertical and horizontal position refers to actually filling block on C matrix
{
    if (ThreadCount == 1){
        for (size_t VerOut = 0; VerOut < Dim1; VerOut += Dim1Part){
            for(size_t Dim2Outer = 0; Dim2Outer < Dim2; Dim2Outer += Dim2Part){
                for(size_t HorOut = 0; HorOut < Dim3; HorOut += Dim3Part){
                    CCInnerParts(VerOut, HorOut, Dim2Outer);
                }
            }
        }
    }
    else{
        StartGuard = std::make_unique<std::latch>(ThreadCount + 1);

        ThreadPackage& Threads = ResourceManager::GetThreads();
        for(unsigned i = 0; i < ThreadCount; ++i){
            Threads.Array[i] = new std::thread(&ThreadInstance<double>, this, &GPMM<double>::CCInnerPartsThreaded);
        }

        unsigned SetupIterations = ThreadCount;
        for (size_t VerOut = 0; VerOut < Dim1; VerOut += Dim1Part){
            if (!CordQue.empty()) continue;

            for(size_t Dim2Outer = 0; Dim2Outer < Dim2; Dim2Outer += Dim2Part){
                for(size_t HorOut = 0; HorOut < Dim3; HorOut += Dim3Part){
                    static constexpr size_t VerInBlockSize = 8;
                    const size_t VerInRange = min(VerOut + Dim1Part, Dim1);

                    for(size_t VerIn = VerOut; VerIn < VerInRange; VerIn += VerInBlockSize){
                        if (SetupIterations--){
                            CordQue.push({VerIn, HorOut, Dim2Outer});

                            if (!SetupIterations) StartGuard->arrive_and_wait();
                        }
                        else{
                            QueGuard.lock();
                            CordQue.push({VerIn, HorOut, Dim2Outer});
                            QueGuard.unlock();
                        }
                    }
                }
            }
        }

        WorkDone = true;
        for(unsigned i = 0; i < ThreadCount; ++i){
            Threads.Array[i]->join();
            delete Threads.Array[i];
        }
        Threads.Release();
    }
}

#pragma clang diagnostic pop