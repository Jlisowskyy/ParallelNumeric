// Author: Jakub Lisowski

#include "../Include/Operations/MatrixMultiplication.hpp"

#include <algorithm>
#include <immintrin.h>

template<>
inline void GPMM<double>::CCKernelXx6(const size_t HorizontalCord, const size_t VerticalCord, const size_t Dim2Off)
// Actual mathematical kernel used to perform all core calculations
// Every single iteration, it loads the entire cache line, from Matrix A into the memory,
// containing 8 doubles (8x64 = 512(cache line length) = 2 * 256(avx register length)),
// what uses 2 avx registers called there VectorPartBuff{Lower|Upper} accordingly to position on A matrix.
// Then multiply them by matching coefficients from matrix B and save them in register accumulators
//                         _    _    _    _    _    _
//    coefficients  -->   |_|  |_|  |_|  |_|  |_|  |_|
//    2 avx vector     _   _    _    _    _    _    _
//         Upper-->   |v| |r|  |r|  |r|  |r|  |r|  |r|  <--  res vectors represents accumulators
//   Two registers    |e| |e|  |e|  |e|  |e|  |e|  |e|  <--  There is 12 of them 6x2
//   to hold vectors  |c| |s|  |s|  |s|  |s|  |s|  |s|  <--
//                    |_| |_|  |_|  |_|  |_|  |_|  |_|  <--
//                     _   _    _    _    _    _    _
//         Lower-->   |v| |r|  |r|  |r|  |r|  |r|  |r|  <--
//                    |e| |e|  |e|  |e|  |e|  |e|  |e|  <--
//                    |c| |s|  |s|  |s|  |s|  |s|  |s|  <--
//                    |_| |_|  |_|  |_|  |_|  |_|  |_|  <--
//
// Kernels iterates through 240 vectors and coefficients storing them in accumulators and saving to memory at end.
// Utilizing all possible registers (2 - vector parts, 2 - coefficients, 12 - result accumulators = 16).
{
    auto GetOnTargetVectUpper = [&](size_t Shift) -> __m256d&{
        return *((__m256d*) (MatC + (HorizontalCord + Shift) * MatCSoL + VerticalCord));
    };

    auto GetOnTargetVectLower = [&](size_t Shift) -> __m256d&{
        return *((__m256d*) (MatC + (HorizontalCord + Shift) * MatCSoL + VerticalCord + AVXInfo::f64Cap));
    };

    static constexpr size_t VectCoefBufferCount { 2 };
    __m256d VectCoefRegisters[VectCoefBufferCount];
    __m256d VectPartRegisterUpper;
    __m256d VectPartRegisterLower;
    __m256d AccRegistersUpper[CCKernelWidth()] { _mm256_setzero_pd() };
    __m256d AccRegistersLower[CCKernelWidth()] { _mm256_setzero_pd() };

    const size_t LoopRange = std::min(Dim2, Dim2Off + Dim2Part);
    for(size_t kk = Dim2Off; kk < LoopRange; ++kk){
        const double* UPtr = MatA + kk * MatASoL + VerticalCord;
        const double* LPtr = MatA + kk * MatASoL + VerticalCord + AVXInfo::f64Cap;

        VectPartRegisterUpper = _mm256_load_pd(UPtr);
        VectPartRegisterLower = _mm256_load_pd(LPtr);

        // Prefetch next entire line into cache
#ifndef __clang__
        __builtin_prefetch(UPtr + MatASoL);
        __builtin_prefetch(LPtr + MatASoL);
#else
        // TODO: example usage not working proeprly yet
        _mm_prefetch(UPtr + MatASoL, 1);
        _mm_prefetch(LPtr + MatASoL, 0);
#endif

        auto SingleLoadAccumulateOp = [&](size_t Offset) -> void{
            VectCoefRegisters[0] = _mm256_set1_pd(MatB[HorizontalCord * MatBSoL + Dim2Off + Offset * MatBSoL]);
            VectCoefRegisters[1] = _mm256_set1_pd(MatB[HorizontalCord * MatBSoL + Dim2Off + (Offset + 1) * MatBSoL]);

            AccRegistersUpper[Offset] = _mm256_fmadd_pd(VectPartRegisterUpper, VectCoefRegisters[0], AccRegistersUpper[Offset]);
            AccRegistersLower[Offset] = _mm256_fmadd_pd(VectPartRegisterLower, VectCoefRegisters[0], AccRegistersLower[Offset]);
            AccRegistersUpper[Offset + 1] = _mm256_fmadd_pd(VectPartRegisterUpper, VectCoefRegisters[1], AccRegistersUpper[Offset + 1]);
            AccRegistersLower[Offset + 1] = _mm256_fmadd_pd(VectPartRegisterLower, VectCoefRegisters[1], AccRegistersLower[Offset + 1]);
        };

        SingleLoadAccumulateOp(0);
        SingleLoadAccumulateOp(2);
        SingleLoadAccumulateOp(4);
    }

    // Should be unrolled
    for (size_t i = 0; i < CCKernelWidth(); ++i){
        GetOnTargetVectUpper(i) += AccRegistersUpper[i];
        GetOnTargetVectLower(i) += AccRegistersLower[i];
    }
}

template<>
inline void GPMM<double>::CCKernelXxY(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off, size_t HorKernelSize)
    // Same as upper, but there are fewer coefficients. Used to cleaning after the main kernel
{
    auto GetOnTargetVectUpper = [&](size_t Shift) -> __m256d&{
        return *((__m256d*) (MatC + (HorizontalCord + Shift) * MatCSoL + VerticalCord));
    };

    auto GetOnTargetVectLower = [&](size_t Shift) -> __m256d&{
        return *((__m256d*) (MatC + (HorizontalCord + Shift) * MatCSoL + VerticalCord + AVXInfo::f64Cap));
    };

    __m256d VectPartBuffUpper;
    __m256d VectPartBuffLower;
    __m256d VectCoefBuff;

    // Distance between Upper and Lower buffer of the corresponding vector
    static constexpr size_t MaximalAmountOfRegistersNeeded = (CCKernelWidth() - 1);
    __m256d AccRegistersUpper[MaximalAmountOfRegistersNeeded] = { _mm256_setzero_pd() };
    __m256d AccRegistersLower[MaximalAmountOfRegistersNeeded] = { _mm256_setzero_pd() };

    const size_t LoopRange = std::min(Dim2, Dim2Off + Dim2Part);
    for(size_t kk = Dim2Off; kk < LoopRange; ++kk){
        const double* VectPartUpperPtr = MatA + kk * MatASoL + VerticalCord;
        const double* VectPartLowerPtr = MatA + kk * MatASoL + VerticalCord + AVXInfo::f64Cap;

        VectPartBuffUpper = _mm256_load_pd(VectPartUpperPtr);
        VectPartBuffLower = _mm256_load_pd(VectPartLowerPtr);
#ifndef __clang__
        __builtin_prefetch(VectPartUpperPtr + MatASoL);
        __builtin_prefetch(VectPartLowerPtr + MatASoL);
#else
        // TODO: example usage not working proeprly yet
        _mm_prefetch(VectPartUpperPtr + MatASoL, 1);
        _mm_prefetch(VectPartLowerPtr + MatASoL, 0);
#endif

        for (size_t i = 0; i < HorKernelSize; ++i ){
            VectCoefBuff = _mm256_set1_pd(MatB[(HorizontalCord + i) * MatBSoL + kk]);

            AccRegistersUpper[i] = _mm256_fmadd_pd(VectPartBuffUpper, VectCoefBuff, AccRegistersUpper[i]);
            AccRegistersLower[i] = _mm256_fmadd_pd(VectPartBuffLower, VectCoefBuff, AccRegistersLower[i]);
        }
    }

    for (size_t i = 0; i < HorKernelSize; ++i ){
        GetOnTargetVectUpper(i) += AccRegistersUpper[i];
        GetOnTargetVectLower(i) += AccRegistersLower[i];
    }
}

template<>
inline void GPMM<double>::CCInnerParts(const size_t VerOut, const size_t HorOut, const size_t Dim2Outer)
    // Was used in omp testings but now work well in single thread execution
{
    const size_t HorInMaxRange = std::min(HorOut + Dim3Part, Dim3);
    const size_t HorInFullyBlockedRange = (HorInMaxRange / CCKernelWidth() ) * CCKernelWidth();
    const size_t VerInRange = std::min(VerOut + Dim1Part, Dim1);

    #pragma omp parallel for
    for(size_t VerIn = VerOut; VerIn < VerInRange; VerIn += CCKernelHeight()){
        for(size_t HorIn = HorOut; HorIn < HorInFullyBlockedRange; HorIn += CCKernelWidth()){
            CCKernelXx6(HorIn, VerIn, Dim2Outer);
        }
        if (HorInMaxRange != HorInFullyBlockedRange){
            CCKernelXxY(HorInFullyBlockedRange, VerIn, Dim2Outer, HorInMaxRange - HorInFullyBlockedRange);
        }
    }
}

template<>
inline void GPMM<double>::CCInnerPartsThreaded(size_t VerIn, size_t HorOut, size_t Dim2Outer)
    // Function used to divide work between working threads
{
    const size_t HorInMaxRange = std::min(HorOut + Dim3Part, Dim3);
    const size_t HorInFullyBlockedRange = (HorInMaxRange / CCKernelWidth() ) * CCKernelWidth();

    for(size_t HorIn = HorOut; HorIn < HorInFullyBlockedRange; HorIn += CCKernelWidth()){
       CCKernelXx6(HorIn, VerIn, Dim2Outer);
    }
    if (HorInMaxRange != HorInFullyBlockedRange){
        CCKernelXxY(HorInFullyBlockedRange, VerIn, Dim2Outer, HorInMaxRange - HorInFullyBlockedRange);
    }
}

template<>
void ThreadInstance<double>(GPMM<double>* Target, void (GPMM<double>::*Oper)(size_t, size_t, size_t))
    // Working thread body
{
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
    if (ThreadCount == 1)
    {
        for (size_t VerOut = 0; VerOut < Dim1; VerOut += Dim1Part){
            for(size_t Dim2Outer = 0; Dim2Outer < Dim2; Dim2Outer += Dim2Part){
                for(size_t HorOut = 0; HorOut < Dim3; HorOut += Dim3Part){
                    CCInnerParts(VerOut, HorOut, Dim2Outer);
                }
            }
        }
    }
    else{
        StartGuard = std::make_unique<std::latch>(ThreadCount);

        ThreadPackage& Threads = ResourceManager::GetThreads();
        for(unsigned i = 0; i < ThreadCount; ++i){
            Threads.Array[i] = new std::thread(&ThreadInstance<double>, this, &GPMM<double>::CCInnerPartsThreaded);
        }

        // Loops adds coordinates to queue, from which working threads reads locations to execute kernel
        for (size_t VerOut = 0; VerOut < Dim1; VerOut += Dim1Part){
            for(size_t Dim2Outer = 0; Dim2Outer < Dim2; Dim2Outer += Dim2Part){
                while(!CordQue.empty())
                    // Waits to for all working threads to finish job.
                    // Used to avoid race condition when saving results.
                    // TODO: Find better solution in future
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                for(size_t HorOut = 0; HorOut < Dim3; HorOut += Dim3Part){
                    static constexpr size_t VerInBlockSize = 8;
                    const size_t VerInRange = std::min(VerOut + Dim1Part, Dim1);

                    for(size_t VerIn = VerOut; VerIn < VerInRange; VerIn += VerInBlockSize){
                        QueGuard.lock();
                        CordQue.push({VerIn, HorOut, Dim2Outer});
                        QueGuard.unlock();
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