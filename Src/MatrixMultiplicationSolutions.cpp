//
// Created by Jlisowskyy on 17/08/2023.
//

#include "../Include/Operations/MatrixMultiplicationSolutions.hpp"
#include <omp.h>

#define VECTORCOEF0 Src2[(i + ii) * Src2SizeOfLine + j + jj]
#define VECTORCOEF1 Src2[(i + ii + 1) * Src2SizeOfLine + j + jj]
#define VECTORCOEF2 Src2[(i + ii + 2) * Src2SizeOfLine + j + jj]
#define VECTORCOEF3 Src2[(i + ii + 3) * Src2SizeOfLine + j + jj]
#define AVXLINE *((__m256d*)(Src1 + (j + jj) * Src1SizeOfLine + k + kk))
#define TAR0 *((__m256d*)(Target + (i + ii) * TargetSizeOfLine + k + kk))
#define TAR1 *((__m256d*)(Target + (i + ii + 1) * TargetSizeOfLine + k + kk))
#define TAR2 *((__m256d*)(Target + (i + ii + 2) * TargetSizeOfLine + k + kk))
#define TAR3 *((__m256d*)(Target + (i + ii + 3) * TargetSizeOfLine + k + kk))

template<>
void SimpleMultMachine<double>::ProcBlock(unsigned HorizontalCord, unsigned VerticalCord){
    const unsigned i = HorizontalCord;
    const unsigned k = VerticalCord;

    for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
        for (unsigned ii = 0; ii < BlockSize; ii += 4) {

            for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                __m256d acc0 = _mm256_setzero_pd();
                __m256d acc1 = _mm256_setzero_pd();
                __m256d acc2 = _mm256_setzero_pd();
                __m256d acc3 = _mm256_setzero_pd();

                for (unsigned jj = 0; jj < BlockSize; ++jj) {
                    acc0 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF0),acc0);
                    acc1 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF1),acc1);
                    acc2 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF2),acc2);
                    acc3 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF3),acc3);
                }

                TAR0 = _mm256_add_pd(acc0, TAR0);
                TAR1 = _mm256_add_pd(acc1, TAR1);
                TAR2 = _mm256_add_pd(acc2, TAR2);
                TAR3 = _mm256_add_pd(acc3, TAR3);
            }
        }
    }
}

template<>
void SimpleMultMachine<double>::RecuMM(unsigned HorizotnalCord, unsigned VerticalCord, unsigned Length){
    if (Length != BlockSize){
        Length /= 2;
        RecuMM(HorizotnalCord, VerticalCord, Length);
        RecuMM(HorizotnalCord + Length, VerticalCord + Length, Length);
        RecuMM(HorizotnalCord, VerticalCord + Length, Length);
        RecuMM(HorizotnalCord + Length, VerticalCord, Length);
    }
    else{
        ProcBlock(HorizotnalCord, VerticalCord);
    }
}



template<>
void SimpleMultMachine<double>::kernel(size_t HorizontalCord, size_t VerticalCord, size_t offset)
#define VectPartUpperPtr Src1 + (offset + kk) * Src1SizeOfLine + VerticalCord
#define VectPartLowerPtr Src1 + (offset + kk) * Src1SizeOfLine + VerticalCord + 4
#define LoadVectCoef(shift) Src2[HorizontalCord * Src2SizeOfLine + offset + shift + kk]
#define OnTargetVectUpper(shift) *((__m256d*) (Target + (HorizontalCord + shift) * TargetSizeOfLine + VerticalCord))
#define OnTargetVectLower(shift) *((__m256d*) (Target + (HorizontalCord + shift) * TargetSizeOfLine + VerticalCord + 4))
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

    for(size_t kk = 0; kk < 240; ++kk){
//        __builtin_prefetch(VectPartUpperPtr + Src1SizeOfLine);
//        __builtin_prefetch(VectPartLowerPtr + Src1SizeOfLine);

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
void SimpleMultMachine<double>::L3BLOCKED(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
                                          const unsigned VectorsStartingBlock){
    for (size_t i = 0; i < Src1Rows; i += Src1Rows){ //12240
        for(size_t k = 0; k < Src2Rows; k += 240){
            for(size_t j = 0; j < Src2Cols; j += 1020){
                #pragma omp parallel for
                for(size_t ii = 0; ii < Src2Cols; ii += 8){
                    for(size_t jj = 0; jj < 1020; jj += 6){
                        kernel(jj + j, ii + i, k);
                    }
                }
            }
        }
    }
}


void MultAlgo2_Kernel(){}


template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_EE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
                                                       const unsigned VectorsStartingBlock)
{
    for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += 8) {
        for (unsigned j = 0; j < BlocksPerVector; j += 8)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVectors; k += 8) {
                for (unsigned ii = 0; ii < 8; ii += 2) {

                    for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                        __m256d acc0 = _mm256_setzero_pd();
                        __m256d acc1 = _mm256_setzero_pd();
                        __m256d acc2 = _mm256_setzero_pd();
                        __m256d acc3 = _mm256_setzero_pd();

                        for (unsigned jj = 0; jj < BlockSize; ++jj) {
                            acc0 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF0),acc0);
                            acc1 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF1),acc1);
                            acc2 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF2),acc2);
                            acc3 = _mm256_fmadd_pd(AVXLINE,_mm256_set1_pd(VECTORCOEF3),acc3);
                        }

                        TAR0 = _mm256_add_pd(acc0, TAR0);
                        TAR1 = _mm256_add_pd(acc1, TAR1);
                        TAR2 = _mm256_add_pd(acc2, TAR2);
                        TAR3 = _mm256_add_pd(acc3, TAR3);
                    }
                }
            }
        }
    }
}
#pragma clang diagnostic pop

template<>
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_EN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
                                                       const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
    for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
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

        for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
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
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_NE(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
                                                       const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
    for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
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
                for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
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
void SimpleMultMachine<double>::MultAlgo2_CC_Blocks_NN(const unsigned VectorsBlocks, const unsigned BlocksPerVector, const unsigned BlocksPerBaseVectors,
                                                       const unsigned VectorsStartingBlock)
// AVX READY version of previous algorithm
// 0.06
{
    for (unsigned i = VectorsStartingBlock; i < VectorsBlocks; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < BlocksPerVector; j += ElementsInCacheLine)
            // Next iterations without cleaning
        {
            for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
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
                for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
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

        for (unsigned k = 0; k < BlocksPerBaseVectors; k += ElementsInCacheLine) {
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
            for (unsigned k = BlocksPerBaseVectors; k < Src1Rows; ++k) {
                double acc0 = 0;
                for (unsigned j = BlocksPerVector; j < Src2Rows; ++j) {
                    acc0 += Src1[j * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j];
                }
                Target[(i + ii) * TargetSizeOfLine + k] += acc0;
            }
        }
    }
}

template<>
void SimpleMultMachine<double>::MultAlgo1_CC()
{
    const unsigned BlocksPerRun = (Src1Cols / ElementsInCacheLine) * ElementsInCacheLine;
    const unsigned VerticalBlocks = (Src1Rows / ElementsInCacheLine) * ElementsInCacheLine;
    const unsigned HorizontalBlocks = (Src2Cols / ElementsInCacheLine) * ElementsInCacheLine;

    for (unsigned i = 0; i < HorizontalBlocks; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < VerticalBlocks; j += ElementsInCacheLine) {
            //#define SumOverVectors(vector_offset) Src1[(k + kk) * Src1SizeOfLine + j + jj] * Src2[(i + ii + vector_offset) * Src2SizeOfLine + k + kk];
            //#define SaveAccumulatorsOverVector(vector_offset) Target[(i + ii + vector_offset) * TargetSizeOfLine + j + jj]

            for (unsigned k = 0; k < BlocksPerRun; k += ElementsInCacheLine) {

                for (unsigned ii = 0; ii < ElementsInCacheLine; ii += 2) {
                    for (unsigned jj = 0; jj < ElementsInCacheLine; jj += 4) {
                        __m256d acc[2] = { _mm256_set1_pd(0) };

                    }
                }
            }
        }
    }
}