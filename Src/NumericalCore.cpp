//
// Created by Jlisowskyy on 13/08/2023.
//

#include "../Include/Operations/NumericalCore.hpp"

void MatrixMultThreadExecutionUnit::StartExecution() {
    ThreadPackage& Threads = ResourceManager::GetThreads();
    unsigned i;
    for (i = 0; i < ThreadCount-1; ++i) {
        unsigned Start = i * BlocksPerThread;
        unsigned Stop = (i + 1) * BlocksPerThread;

        Threads.Array[i] = new std::thread(&MatrixMultThreadExecutionUnit::MatrixMultThread, this, Start, Stop);
    }
    unsigned Start = i * BlocksPerThread;
    Threads.Array[i] = new std::thread(&MatrixMultThreadExecutionUnit::MatrixMultThread, this, Start, Blocks);

    for (unsigned j = 0; j < ThreadCount; ++j) {
        Threads.Array[j]->join();
        delete Threads.Array[j];
    }

    Threads.Release();
}

void MatrixMultThreadExecutionUnit::MatrixMultThread(unsigned int StartBlock, unsigned int BorderBlock) {
    Synchronizer.arrive_and_wait();

    (Machine->ProcessBlock)(StartBlock, BorderBlock);

    if (m.try_lock()){ // bullshiet
        if (!FrameDone){
            Machine->ProcessFrame();
            FrameDone = true;
        }

    }
}

// -----------------------------------------
// AVX specialisations
// -----------------------------------------


// -----------------------------------------
// Matrix sum
// -----------------------------------------

#ifdef __AVX__

template<>
void MatrixSumHelperAlignedArrays(double *Target, const double *const Input1, const double *const Input2,
                                  const unsigned long Elements) {
    const auto VectInput1 = (const __m256d* const)Input1;
    const auto VectInput2 = (const __m256d* const)Input2;
    auto VectTarget = (__m256d*)Target;

    const unsigned long VectSize = Elements / DOUBLE_VECTOR_LENGTH;
    for (unsigned long i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_pd(VectInput1[i], VectInput2[i]);
    }

    for (unsigned long i = VectSize * DOUBLE_VECTOR_LENGTH; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

template<>
void MatrixSumHelperAlignedArrays(float *Target, const float *const Input1, const float *const Input2,
                                  const unsigned long Elements) {
    const auto VectInput1 = (const __m256* const)Input1;
    const auto VectInput2 = (const __m256* const)Input2;
    auto VectTarget = (__m256*)Target;

    const unsigned long VectSize = Elements / FLOAT_VECTOR_LENGTH;
    for (unsigned long i = 0; i < VectSize; ++i) {
        VectTarget[i] = _mm256_add_ps(VectInput1[i], VectInput2[i]);
    }

    for (unsigned long i = VectSize * FLOAT_VECTOR_LENGTH; i < Elements; ++i) {
        Target[i] = Input1[i] + Input2[i];
    }
}

#endif // __AVX__

// ---------------------------------------
// Matrix multiplication
// ---------------------------------------

#if defined(__AVX__) && defined(__FMA__)

template<>
void CCTarHor_MultMachine<double>::EEBlocks(const unsigned VectorStartingBlock, const unsigned VectorBlocksBorder)
{
    unsigned Mult = 1;
    for (unsigned i = VectorStartingBlock / Mult; i < VectorBlocksBorder / Mult; i += ElementsInCacheLine * Mult) {
        for (unsigned j = 0; j < BlocksPerVector / Mult; j += ElementsInCacheLine * Mult)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVector / Mult; k += ElementsInCacheLine * Mult) {
                for (unsigned ii = 0; ii < ElementsInCacheLine * Mult; ii += 2) {

                    for (unsigned kk = 0; kk < ElementsInCacheLine * Mult; kk += 4) {
                        __m256d acc0 = _mm256_setzero_pd();
                        __m256d acc1 = _mm256_setzero_pd();

                        for (unsigned jj = 0; jj < ElementsInCacheLine * Mult; ++jj) {
                            double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
                            double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                            __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

                            acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef0),
                                                   acc0
                            );

                            acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef1),
                                                   acc1
                            );
                        }

                        *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                     *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                        );

                        *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                         *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                        );
                    }
                }
            }
        }
    }
}

template<>
void CCTarHor_MultMachine<double>::ENBlocks(const unsigned VectorStartingBlock, const unsigned VectorBlocksBorder)
{
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVector; k += ElementsInCacheLine) {
                for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

                    for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
                        __m256d acc0 = _mm256_set1_pd(0);
                        __m256d acc1 = _mm256_set1_pd(0);

                        for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
                            double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
                            double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                            __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

                            acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef0),
                                                   acc0
                            );

                            acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef1),
                                                   acc1
                            );
                        }

                        *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                     *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                        );

                        *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                         *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                        );
                    }
                }
            }
        }

        for (unsigned k = 0; k < BlocksPerBaseVector; k += ElementsInCacheLine) {
            for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

                for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();

                    for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
                        double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j];
                        double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j];
                        __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + j * Src1SizeOfLine + k + kk));

                        acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                               _mm256_set1_pd(VectorCoef0),
                                               acc0
                        );

                        acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                               _mm256_set1_pd(VectorCoef1),
                                               acc1
                        );
                    }

                    *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                 *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                    );

                    *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                     *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                    );
                }
            }
        }
    }
}


template<>
void CCTarHor_MultMachine<double>::NEBlocks(const unsigned VectorStartingBlock, const unsigned VectorBlocksBorder)
{
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVector; k += ElementsInCacheLine) {
                for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

                    for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
                        __m256d acc0 = _mm256_setzero_pd();
                        __m256d acc1 = _mm256_setzero_pd();

                        for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
                            double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
                            double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                            __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

                            acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef0),
                                                   acc0
                            );

                            acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef1),
                                                   acc1
                            );
                        }

                        *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                     *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                        );

                        *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                         *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                        );
                    }
                }
            }

            for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
                for (unsigned k = BlocksPerBaseVector; k < Src1Rows; ++k) {
                    double acc0 = 0;
                    double acc1 = 0;
                    double acc2 = 0;
                    double acc3 = 0;

                    for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j + jj];
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
                    }


                    Target[(i + ii) * TargetSizeOfLine + k] += acc0;
                    Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
                    Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
                    Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

                }
            }
        }
    }
}

template<>
void CCTarHor_MultMachine<double>::NNBlocks(const unsigned VectorStartingBlock, const unsigned VectorBlocksBorder)
{
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVector; k += ElementsInCacheLine) {
                for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

                    for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
                        __m256d acc0 = _mm256_setzero_pd();
                        __m256d acc1 = _mm256_setzero_pd();

                        for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
                            double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j + jj];
                            double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                            __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk));

                            acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef0),
                                                   acc0
                            );

                            acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                                   _mm256_set1_pd(VectorCoef1),
                                                   acc1
                            );
                        }

                        *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                     *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                        );

                        *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                         *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                        );
                    }
                }
            }

            for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 4) {
                for (unsigned k = BlocksPerBaseVector; k < Src1Rows; ++k) {
                    double acc0 = 0;
                    double acc1 = 0;
                    double acc2 = 0;
                    double acc3 = 0;

                    for (unsigned jj = 0; jj < ElementsInCacheLine; ++jj) {
                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j + jj];
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
                    }


                    Target[(i + ii) * TargetSizeOfLine + k] += acc0;
                    Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
                    Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
                    Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

                }
            }
        }

        for (unsigned k = 0; k < BlocksPerBaseVector; k += ElementsInCacheLine) {
            for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {

                for (unsigned kk = 0; kk < ElementsInCacheLine; kk += 4) {
                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();

                    for (unsigned j = BlocksPerVector; j < Src1Cols; ++j) {
                        double VectorCoef0 = Src2[(i + ii) * Src2SizeOfLine + j];
                        double VectorCoef1 = Src2[(i + ii + 1) * Src2SizeOfLine + j];
                        __m256d BaseVectorAVXLine = *((__m256d*)(Src1 + j * Src1SizeOfLine + k + kk));

                        acc0 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                               _mm256_set1_pd(VectorCoef0),
                                               acc0
                        );

                        acc1 = _mm256_fmadd_pd(BaseVectorAVXLine,
                                               _mm256_set1_pd(VectorCoef1),
                                               acc1
                        );
                    }

                    *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc0,
                                                                                                 *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
                    );

                    *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk)) = _mm256_add_pd(acc1,
                                                                                                     *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
                    );
                }
            }
        }

        for (unsigned ii = 0; ii < ElementsInCacheLine; ++ii) {
            for (unsigned k = BlocksPerBaseVector; k < Src1Rows; ++k) {
                double acc0 = 0;
                for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
                    acc0 += Src1[j * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j];
                }
                Target[(i + ii) * TargetSizeOfLine + k] += acc0;
            }
        }
    }
}
#endif // __AVX__ __FMA__

// ------------------------------------------
// Dot product
// ------------------------------------------

#ifdef __AVX__

template<>
double DotProduct(double *const Src1, double *const Src2, unsigned long Range) {
    auto VectSrc1 = (__m256d* const) Src1;
    auto VectSrc2 = (__m256d* const) Src2;
    __m256d Store = _mm256_set_pd(0, 0, 0, 0);

    const unsigned long VectRange = Range/4;
    for (unsigned long i = 0; i < VectRange; ++i) {
        Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
    }

    double EndResult = 0;
    for (unsigned long i = VectRange * 4; i < Range; ++i) {
        EndResult += Src1[i] * Src2[i];
    }

    auto result =(double*) &Store;
    return result[0] + result[1] + result[2] + result[3] + EndResult;
}

#endif // __AVX__

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* const Src1, const double* const Src2, const unsigned Threads, const unsigned long Range) :
        DPMCore<double>(Src1, Src2, Threads, Range, (Range / (Threads * DOUBLE_VECTOR_LENGTH)) * Threads * DOUBLE_VECTOR_LENGTH),
        ElemPerThread{ Range / (Threads * DOUBLE_VECTOR_LENGTH) }
{}

template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* const Src1, const float* const Src2, const unsigned Threads, const unsigned long Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (Range / (Threads * FLOAT_VECTOR_LENGTH)) * Threads * FLOAT_VECTOR_LENGTH),
        ElemPerThread{ Range / (Threads * FLOAT_VECTOR_LENGTH) }
{}


template<>
void DotProductMachineChunked<double>::StartThread(const unsigned ThreadID) {
    const auto VectSrc1 = (const __m256d* const) Src1;
    const auto VectSrc2 = (const __m256d* const) Src2;
    __m256d Store = _mm256_set_pd(0, 0, 0, 0);
    const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;

    Counter.arrive_and_wait();
    for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
        Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
    }

    WriteCounter.arrive_and_wait();
    auto Result = (double*) &Store;
    ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3];
}

template<>
void DotProductMachineChunked<float>::StartThread(const unsigned ThreadID) {
    Counter.arrive_and_wait();
    auto VectSrc1 = (__m256* const) Src1;
    auto VectSrc2 = (__m256* const) Src2;
    __m256 Store = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

    const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;
    for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
        Store = _mm256_fmadd_ps(VectSrc1[i], VectSrc2[i], Store);
    }

    WriteCounter.arrive_and_wait();
    auto Result = (float*)&Store;
    ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3] + Result[4] + Result[5] + Result[6] + Result[7];
}

#define PER_ITERATION_DOUBLE 2
#define PER_CIRCLE_FLOAT 2

template<>
DotProductMachineComb<double>::DotProductMachineComb(const double* const Src1, const double* const Src2, const unsigned Threads, const unsigned long Range) :
        DPMCore<double>(Src1, Src2, Threads, Range,
                        (((Range / DOUBLE_VECTOR_LENGTH) * DOUBLE_VECTOR_LENGTH) / PER_ITERATION_DOUBLE) * (CACHE_LINE / PER_ITERATION_DOUBLE)), LoopRange{Range / DOUBLE_VECTOR_LENGTH },
        PerCircle{PER_ITERATION_DOUBLE }
{}

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* const Src1, const float* const Src2, const unsigned Threads, const unsigned long Range) :
        DPMCore<float>(Src1, Src2, Threads, Range, (((Range / FLOAT_VECTOR_LENGTH)* FLOAT_VECTOR_LENGTH) / PER_CIRCLE_FLOAT)* PER_CIRCLE_FLOAT), LoopRange{ Range / FLOAT_VECTOR_LENGTH },
        PerCircle{ PER_CIRCLE_FLOAT }
{}

#define SingleOP_D(offset) Store = _mm256_fmadd_pd(VectSrc1[i+offset], VectSrc2[i+offset], Store)
#define SingleOP_F(offset) Store =_mm256_fmadd_ps(VectSrc1[i+offset], VectSrc2[i+offset], Store)

template<>
void DotProductMachineComb<double>::StartThread(const unsigned ThreadID)
{
    const __m256d* const VectSrc1 = ((__m256d* const) Src1) + PerCircle * ThreadID;
    const __m256d* const VectSrc2 = ((__m256d* const) Src2) + PerCircle * ThreadID;
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
    const __m256* const VectSrc1 = ((__m256* const) Src1) + PerCircle * ThreadID;
    const __m256* const VectSrc2 = ((__m256* const) Src2) + PerCircle * ThreadID;
    __m256 Store = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

    const unsigned long Jump = Threads * PerCircle;
    Counter.arrive_and_wait();

    for (unsigned long i = 0; i < LoopRange; i += Jump) {
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