//
// Created by Jlisowskyy on 13/08/2023.
//

#include "../Include/Operations/NumericalCore.hpp"

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
double DotProduct(const double *const Src1, const double *const Src2, const size_t Range) {
    const size_t VectRange = (Range / 32) * 32;
    __m256d Acc0 = _mm256_setzero_pd();
    __m256d Acc1 = _mm256_setzero_pd();
    __m256d Acc2 = _mm256_setzero_pd();
    __m256d Acc3 = _mm256_setzero_pd();
    __m256d Acc4 = _mm256_setzero_pd();
    __m256d Acc5 = _mm256_setzero_pd();
    __m256d Acc6 = _mm256_setzero_pd();
    __m256d Acc7 = _mm256_setzero_pd();

    for (size_t i = 0; i < VectRange; i+=32) {
        Acc0 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i),
                _mm256_load_pd(Src2 + i),
                Acc0
        );
        Acc1 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 4),
                _mm256_load_pd(Src2 + i + 4),
                Acc1
        );
        Acc2 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 8),
                _mm256_load_pd(Src2 + i + 8),
                Acc2
        );
        Acc3 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 12),
                _mm256_load_pd(Src2 + i + 12),
                Acc3
        );
        Acc4 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 16),
                _mm256_load_pd(Src2 + i + 16),
                Acc4
        );
        Acc5 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 20),
                _mm256_load_pd(Src2 + i + 20),
                Acc5
        );
        Acc6 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 24),
                _mm256_load_pd(Src2 + i + 24),
                Acc6
        );
        Acc7 = _mm256_fmadd_pd(
                _mm256_load_pd(Src1 + i + 28),
                _mm256_load_pd(Src2 + i + 28),
                Acc7
        );
    }

#define HorSumAVX(avx_double) ((double*)&avx_double)[0] + ((double*)&avx_double)[1] + ((double*)&avx_double)[2] + ((double*)&avx_double)[3];
    double RetVal = HorSumAVX(Acc0) + HorSumAVX(Acc1) + HorSumAVX(Acc2) + HorSumAVX(Acc3) + HorSumAVX(Acc4) + HorSumAVX(Acc5) + HorSumAVX(Acc6) + HorSumAVX(Acc7);
    for (size_t i = VectRange; i < Range; ++i) {
        RetVal += Src1[i] * Src2[i];
    }

    return RetVal;
}

#endif // __AVX__

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* const Src1, const double* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<double>(Src1, Src2, Threads, Range, (Range / (Threads * AVXInfo::f64Cap)) * Threads * AVXInfo::f64Cap),
        ElemPerThread{ Range / (Threads * AVXInfo::f64Cap) }
{}

template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* const Src1, const float* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (Range / (Threads * AVXInfo::f64Cap)) * Threads * AVXInfo::f64Cap),
        ElemPerThread{ Range / (Threads * AVXInfo::f32Cap) }
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
                        (((Range / AVXInfo::f64Cap) * AVXInfo::f64Cap) / PER_ITERATION_DOUBLE) * (CacheInfo::LineSize / PER_ITERATION_DOUBLE)), LoopRange{Range / AVXInfo::f64Cap },
        PerCircle{PER_ITERATION_DOUBLE }
{}

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* const Src1, const float* const Src2, const unsigned Threads, const size_t Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (((Range / AVXInfo::f32Cap) * AVXInfo::f32Cap) / PER_CIRCLE_FLOAT) * PER_CIRCLE_FLOAT), LoopRange{Range / AVXInfo::f32Cap },
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