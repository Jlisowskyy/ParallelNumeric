//
// Created by Jlisowskyy on 13/08/2023.
//

#include "../Include/Operations/NumericalCore.hpp"
#include "../Include/Maintenance/Debuggers.hpp"

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

    const unsigned long VectSize = Elements / DOUBLE_VECTOR_LENGTH;
    for (size_t i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_pd(VectInput1[i], VectInput2[i]);
    }

    for (size_t i = VectSize * DOUBLE_VECTOR_LENGTH; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

template<>
void MatrixSumHelperAlignedArrays(float *Target, const float *const Input1, const float *const Input2,
                                  const size_t Elements) {
    const auto VectInput1 = (const __m256*)Input1;
    const auto VectInput2 = (const __m256*)Input2;
    auto VectTarget = (__m256*)Target;

    const size_t VectSize = Elements / SINGLE_VECTOR_LENGTH;
    for (size_t i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_ps(VectInput1[i], VectInput2[i]);
    }

    for (size_t i = VectSize * SINGLE_VECTOR_LENGTH; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

#endif // __AVX__

// ---------------------------------------
// Matrix multiplication
// ---------------------------------------

#if defined(__AVX__) && defined(__FMA__)

#endif // __AVX__ __FMA__

// ------------------------------------------
// Dot product
// ------------------------------------------

#ifdef __AVX__

template<>
double DotProduct(double *const Src1, double *const Src2, size_t Range) {
    const auto VectSrc1 = (__m256d*) Src1;
    const auto VectSrc2 = (__m256d*) Src2;
    __m256d Store = _mm256_set_pd(0, 0, 0, 0);

    const size_t VectRange = Range/4;
    for (size_t i = 0; i < VectRange; ++i) {
        Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
    }

    double EndResult = 0;
    for (size_t i = VectRange * 4; i < Range; ++i) {
        EndResult += Src1[i] * Src2[i];
    }

    auto result =(double*) &Store;
    return result[0] + result[1] + result[2] + result[3] + EndResult;
}

#endif // __AVX__

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* const Src1, const double* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<double>(Src1, Src2, Threads, Range, (Range / (Threads * DOUBLE_VECTOR_LENGTH)) * Threads * DOUBLE_VECTOR_LENGTH),
        ElemPerThread{ Range / (Threads * DOUBLE_VECTOR_LENGTH) }
{}

template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* const Src1, const float* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (Range / (Threads * SINGLE_VECTOR_LENGTH)) * Threads * SINGLE_VECTOR_LENGTH),
        ElemPerThread{ Range / (Threads * SINGLE_VECTOR_LENGTH) }
{}


template<>
void DotProductMachineChunked<double>::StartThread(const unsigned ThreadID) {
    const auto VectSrc1 = (const __m256d*) Src1;
    const auto VectSrc2 = (const __m256d*) Src2;
    __m256d Store = _mm256_set_pd(0, 0, 0, 0);
    const size_t LoopRange = (ThreadID + 1) * ElemPerThread;

    Counter.arrive_and_wait();
    for (size_t i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
        Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
    }

    WriteCounter.arrive_and_wait();
    auto Result = (double*) &Store;
    ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3];
}

template<>
void DotProductMachineChunked<float>::StartThread(const unsigned ThreadID) {
    Counter.arrive_and_wait();
    const auto VectSrc1 = (__m256*) Src1;
    const auto VectSrc2 = (__m256*) Src2;
    __m256 Store = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

    const size_t LoopRange = (ThreadID + 1) * ElemPerThread;
    for (size_t i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
        Store = _mm256_fmadd_ps(VectSrc1[i], VectSrc2[i], Store);
    }

    WriteCounter.arrive_and_wait();
    auto Result = (float*)&Store;
    ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3] + Result[4] + Result[5] + Result[6] + Result[7];
}

#define PER_ITERATION_DOUBLE 2
#define PER_CIRCLE_FLOAT 2

template<>
DotProductMachineComb<double>::DotProductMachineComb(const double* const Src1, const double* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<double>(Src1, Src2, Threads, Range,
                        (((Range / DOUBLE_VECTOR_LENGTH) * DOUBLE_VECTOR_LENGTH) / PER_ITERATION_DOUBLE) * (CACHE_LINE / PER_ITERATION_DOUBLE)), LoopRange{Range / DOUBLE_VECTOR_LENGTH },
        PerCircle{PER_ITERATION_DOUBLE }
{}

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* const Src1, const float* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (((Range / SINGLE_VECTOR_LENGTH) * SINGLE_VECTOR_LENGTH) / PER_CIRCLE_FLOAT) * PER_CIRCLE_FLOAT), LoopRange{Range / SINGLE_VECTOR_LENGTH },
        PerCircle{ PER_CIRCLE_FLOAT }
{}

#define SingleOP_D(offset) Store = _mm256_fmadd_pd(VectSrc1[i+offset], VectSrc2[i+offset], Store)
#define SingleOP_F(offset) Store =_mm256_fmadd_ps(VectSrc1[i+offset], VectSrc2[i+offset], Store)

template<>
void DotProductMachineComb<double>::StartThread(const unsigned ThreadID)
{
    const auto VectSrc1 = ((const __m256d*) Src1) + PerCircle * ThreadID;
    const auto VectSrc2 = ((const __m256d*) Src2) + PerCircle * ThreadID;
    __m256d Store = _mm256_set_pd(0, 0, 0, 0);

    const unsigned long Jump = Threads * PerCircle;
    Counter.arrive_and_wait();

    for (unsigned long i = 0; i < LoopRange; i += Jump) {
        SingleOP_D(0);
        SingleOP_D(1);
    }

    WriteCounter.arrive_and_wait();

    double ret = 0;
    auto result = (double*)&Store;

    for (int i = 0; i < 4; ++i) {
        ret += result[i];
    }

    ResultArray[ThreadID] = ret;
}


template<>
void DotProductMachineComb<float>::StartThread(const unsigned ThreadID)
{
    const auto VectSrc1 = ((const __m256*) Src1) + PerCircle * ThreadID;
    const auto VectSrc2 = ((const __m256*) Src2) + PerCircle * ThreadID;
    __m256 Store = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

    const size_t Jump = Threads * PerCircle;
    Counter.arrive_and_wait();

    for (size_t i = 0; i < LoopRange; i += Jump) {
        SingleOP_F(0);
        SingleOP_F(1);
    }

    WriteCounter.arrive_and_wait();

    float RetVal = 0;
    auto RetP = (float*)&Store;

    for (int i = 0; i < 4; ++i) {
        RetVal += RetP[i];
    }

    ResultArray[ThreadID] = RetVal;
}

#endif // __AVX__ __FMA__

// ------------------------------------------
// Outer product Implementation
// ------------------------------------------

