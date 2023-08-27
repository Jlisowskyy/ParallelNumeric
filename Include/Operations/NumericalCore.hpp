
// Author: Jakub Lisowski

#ifndef PARALLELNUM_HELPERS_H_
#define PARALLELNUM_HELPERS_H_

#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <latch>
#include <cstdlib>
#include <algorithm>
#include <mutex>

#include "../Management/ResourceManager.hpp"
//#include "MatrixMultiplicationSolutions.hpp"

// ------------------------------------------
// Sum of matrices functions
// ------------------------------------------

template<typename NumType>
void MatrixSumHelperAlignedArrays(NumType*Target, const NumType* Input1, const NumType* Input2, size_t Elements);
// Function used to Sum matrices

#ifdef __AVX__
//defined(__AVX__) && defined(__FMA__)
template<>
void MatrixSumHelperAlignedArrays(double* Target,const double* const Input1, const double* const Input2, const size_t Elements);

template<>
void MatrixSumHelperAlignedArrays(float* Target, const float* const Input1, const float* const Input2, const size_t Elements);

#endif // __AVX__

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  size_t StartCol, size_t StopCol, size_t Rows,
                                                  size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  size_t StartRow, size_t StopRow, size_t Cols,
                                                  size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  size_t StartCol, size_t StopCol, size_t Rows,
                                                  size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  size_t StartRow, size_t StopRow, size_t Cols,
                                                  size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        size_t StartCol, size_t StopCol, size_t Rows,
                                                        size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        size_t StartRow, size_t StopRow, size_t Cols,
                                                        size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        size_t StartRow, size_t StopRow, size_t Cols,
                                                        size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        size_t StartCol, size_t StopCol, size_t Rows,
                                                        size_t TargetSizeOfLine, size_t Input1SizeOfLine, size_t Input2SizeOfLine);

//#define I2A(offset) Input2[(i + z + offset) * Input2SizeOfLine + (j + k)]
//
//template<>
//void MatrixSumHelperNotAlignedArrays_CR(double* Target, double* Input1, double* Input2, unsigned StartCol, unsigned StopCol, unsigned Rows,
//	unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine) {
//
//	unsigned BlockSize = CACHE_LINE / sizeof(double);
//	unsigned VectorsInCacheLine = 2;
//
//	for (unsigned i = 0; i < Rows; i += BlockSize) {
//		for (unsigned j = StartCol; j < StopCol; j += BlockSize) {
//			for (unsigned k = 0; k < BlockSize; ++k) {
//				__m256d* VectTarget = (__m256d*) (Target + (j + k) * TargetSizeOfLine + i);
//				__m256d* VectInput1 = (__m256d*) (Input1 + (j + k) * Input1SizeOfLine + i);
//
//				for (unsigned z = 0; z < VectorsInCacheLine; ++z) {
//					VectTarget[z] = _mm256_add_pd(VectInput1[z],
//						_mm256_set_pd(I2A(0), I2A(1), I2A(2), I2A(3)));
//				}
//			}
//		}
//	}
//}

// ------------------------------------------
// Matrix transposition solutions
// ------------------------------------------

// Naive solution
template<typename NumType>
void TransposeMatrixRowStored(NumType* Dst, NumType* Src, size_t SrcLines, size_t SrcElementsPerLine,
                              size_t DstSizeOfLine, size_t SrcSizeOfLine);

// ------------------------------------------
// Dot product code
// ------------------------------------------

template<typename NumType>
NumType DotProduct(NumType* Src1, NumType* Src2, size_t Range);

#ifdef __AVX__

template<>
double DotProduct(double * Src1, double * Src2, size_t Range);

#endif // __AVX__

// ------------------------------------------
// Solution 1
// ------------------------------------------

template<typename NumType>
class DPMCore{
protected:
	const NumType* const Src1;
	const NumType* const Src2;
	const unsigned Threads;
	const size_t Range;
	const size_t EndIndex;
	NumType ResultArray[MaxCPUThreads] = {NumType() };
	std::latch Counter;
	std::latch WriteCounter;
public:
	DPMCore(const NumType* const Src1, const NumType* const Src2, unsigned Threads, size_t Range, size_t EndIndex) :
		Src1{ Src1 }, Src2{ Src2 }, Threads{ Threads }, Range{ Range }, EndIndex{ EndIndex },
		Counter{ Threads }, WriteCounter{ Threads }
	{}

	NumType GetResult();
};

// ------------------------------------------
// Solution 2
// ------------------------------------------

template<typename NumType>
class DotProductMachineChunked: public DPMCore<NumType> {
	const size_t ElemPerThread;
public:
	DotProductMachineChunked(const NumType* const Src1, const NumType* const Src2, unsigned Threads, size_t Range) :
		ElemPerThread{ Range / (size_t) Threads }, DPMCore<NumType>(Src1, Src2, Threads, Range, (Range / (size_t)Threads) * (size_t)Threads)
	{}

	void StartThread(unsigned ThreadID);
};

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* Src1, const double* Src2, unsigned Threads, size_t Range);

template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* Src1, const float* Src2, unsigned Threads, size_t Range);

template<>
void DotProductMachineChunked<double>::StartThread(unsigned ThreadID);

template<>
void DotProductMachineChunked<float>::StartThread(unsigned ThreadID);

#endif

template<typename NumType>
class DotProductMachineComb: public DPMCore<NumType> {
	const size_t LoopRange;
	const size_t PerCircle = CACHE_LINE / sizeof(NumType);
public:
	DotProductMachineComb(const NumType* const Src1, const NumType* const Src2, unsigned Threads, size_t Range) :
		LoopRange{ Range }, DPMCore<NumType>(Src1, Src2, Threads, Range, (Range / (CACHE_LINE / sizeof(NumType))) * (CACHE_LINE / sizeof(NumType)))
	{}

	void StartThread(unsigned ThreadID);
};

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineComb<double>::DotProductMachineComb(const double* Src1, const double* Src2, unsigned Threads, size_t Range);

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* Src1, const float* Src2, unsigned Threads, size_t Range);

template<>
void DotProductMachineComb<double>::StartThread(unsigned ThreadID);

template<>
void DotProductMachineComb<float>::StartThread(unsigned ThreadID);

#endif

// ------------------------------------------
// Outer Product
// ------------------------------------------

template<typename NumType>
class OPM
    // Outer Product Machine
    // TODO Optimize for L3 - long vectors
{
    static constexpr size_t ElementsPerCacheLine = CACHE_LINE / sizeof(NumType);

    const NumType* CoefPtr;
    const NumType* VectPtr;
    NumType* const MatC;
    size_t CoefSize;
    size_t VectSize;
    const size_t MatCSoL;
public:
    OPM(const NumType* VectA, const NumType* VectB, NumType* MatC, size_t ASize,
        size_t BSize, size_t MatCSoL, bool IsHor = false);
    inline void Perform();
};

// ------------------------------------------
// Vector X Matrix Multiplication
// ------------------------------------------

template<typename NumType>
class VMM{
    const NumType* const MatA;
    const NumType* const VectB;
    NumType* const VectC;

    const size_t MatARows;
    const size_t MatACols;
    const size_t MatASoL;
    const bool IsMatHor;
    void RMVKernel12x4(size_t HorizontalCord, size_t VerticalCord) {};
    void CMVKernel12x4(size_t HorizontalCord, size_t VerticalCord) {};
    inline void PerformCVM();
    inline void PerformRVM() {};
    inline void PerformCMV();
    inline void PerformRMV();
public:
    VMM(const NumType* MatA, const NumType* VectB, NumType* VectC, size_t MatARows, size_t MatACols, size_t MatASoL, bool IsHor);
    void PerformVM(){
        if (IsMatHor) PerformRVM();
        else PerformCVM();
    }
    void PerformMV(){
        if (IsMatHor) PerformRMV();
        else PerformCMV();
    }
};

#if defined(__AVX__) && defined(__FMA__)

template<>
void VMM<double>::PerformCMV();

template<>
void VMM<double>::PerformRMV();

template<>
void VMM<double>::RMVKernel12x4(size_t HorizontalCord, size_t VerticalCord);

template<>
void VMM<double>::CMVKernel12x4(size_t HorizontalCord, size_t VerticalCord);

#endif

// ------------------------------------------
// Matrix Sum Implementation
// ------------------------------------------

template<typename T>
void MatrixSumHelperAlignedArrays(T *const Target, const T *const Input1, const T *const Input2,
                                  const size_t Elements) {
    for (unsigned long i = 0; i < Elements; ++i)
        Target[i] = Input1[i] + Input2[i];
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                                  const size_t StartCol, const size_t StopCol,
                                                  const size_t Rows, const size_t TargetSizeOfLine,
                                                  const size_t Input1SizeOfLine,
                                                  const size_t Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t RowsRange = ((size_t)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (size_t j = StartCol; j < StopCol; j += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(i + z) * TargetSizeOfLine + j + k] = Input1[(i + z) * Input1SizeOfLine + j + k] + Input2[(j + k) * Input2SizeOfLine + i + z];
                }
            }
        }
    }

    for (size_t i = StartCol; i < StopCol; ++i) {
        for (size_t j = RowsRange; j < Rows; ++j) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                Target[j * TargetSizeOfLine + i + k] = Input1[j * Input1SizeOfLine + i + k] + Input2[(i + k) * Input2SizeOfLine + j];
            }
        }
    }
}

template<typename NumType>
void
MatrixSumHelperNotAlignedArrays_RC_DivByRows(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                             const size_t StartRow, const size_t StopRow,
                                             const size_t Cols, const size_t TargetSizeOfLine,
                                             const size_t Input1SizeOfLine, const size_t Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t j = StartRow; j < StopRow; j += ElementsInCacheLine) {
        for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(j + k) * TargetSizeOfLine + i + z] = Input1[(j + k) * Input1SizeOfLine + i + z] + Input2[(i + z) * Input2SizeOfLine + j + k];
                }
            }
        }
    }

    for (unsigned j = StartRow; j < StopRow; ++j) {
        for (unsigned i = ColsRange; i < Cols; ++i) {
            Target[j * TargetSizeOfLine + i] = Input1[j * Input1SizeOfLine + i] + Input2[i * Input2SizeOfLine + j];
        }
    }

}

template<typename NumType>
void
MatrixSumHelperNotAlignedArrays_CR_DivByCols(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                             const size_t StartCol, const size_t StopCol,
                                             const size_t Rows, const size_t TargetSizeOfLine,
                                             const size_t Input1SizeOfLine, const size_t Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t RowsRange = ((size_t) (Rows / ElementsInCacheLine)) * ElementsInCacheLine;
    for (size_t j = StartCol; j < StopCol; j += ElementsInCacheLine) {
        for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(j + k) * TargetSizeOfLine + i + z] = Input1[(j + k) * Input1SizeOfLine + i + z] + Input2[(i + z) * Input2SizeOfLine + j + k];
                }
            }
        }
    }

    for (size_t j = StartCol; j < StopCol; ++j) {
        for (size_t z = RowsRange; z < Rows; ++z) {
            Target[j * TargetSizeOfLine + z] = Input1[j * Input1SizeOfLine + z] + Input2[z * Input2SizeOfLine + j];
        }
    }
}

template<typename NumType>
void
MatrixSumHelperNotAlignedArrays_CR_DivByRows(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                             const size_t StartRow, const size_t StopRow,
                                             const size_t Cols, const size_t TargetSizeOfLine,
                                             const size_t Input1SizeOfLine, const size_t Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = StartRow; j < StopRow; j += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(i + k) * TargetSizeOfLine + j + z] = Input1[(i + k) * Input1SizeOfLine + j + z] + Input2[(j + z) * Input2SizeOfLine + i + k];
                }
            }
        }
    }

    for (size_t i = StartRow; i < StopRow; i += ElementsInCacheLine) {
        for (size_t z = ColsRange; z < Cols; ++z) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k)
                Target[z * TargetSizeOfLine + i + k] = Input1[z * Input1SizeOfLine + i + k] + Input2[(i + k) * Input2SizeOfLine + z];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const size_t StartCol,
                                                        const size_t StopCol, const size_t Rows,
                                                        const size_t TargetSizeOfLine,
                                                        const size_t Input1SizeOfLine,
                                                        const size_t Input2SizeOfLine) {
    if (StartCol == StopCol) return;

    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t RowsRange = ((size_t)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (size_t k = StartCol; k < StopCol; ++k) {
            for (size_t j = 0; j < ElementsInCacheLine; ++j) {
                Target[k * TargetSizeOfLine + i + j] = Input1[k * Input1SizeOfLine + i + j] + Input2[(i + j) * Input2SizeOfLine + k];
            }
        }
    }

    for (size_t i = RowsRange; i < Rows; ++i) {
        for (size_t k = StartCol; k < StopCol; ++k) {
            Target[k * TargetSizeOfLine + i] = Input1[k * Input1SizeOfLine + i] + Input2[i * Input2SizeOfLine + k];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const size_t StartRow,
                                                        const size_t StopRow, const size_t Cols,
                                                        const size_t TargetSizeOfLine,
                                                        const size_t Input1SizeOfLine,
                                                        const size_t Input2SizeOfLine) {
    if (StartRow == StopRow) return;

    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = 0; j < ElementsInCacheLine; ++j) {
            for (size_t k = StartRow; k < StopRow; ++k) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (size_t i = ColsRange; i < Cols; ++i) {
        for (size_t k = StartRow; k < StopRow; ++k) {
            Target[i * TargetSizeOfLine + k] = Input1[i * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame(NumType *Target, const NumType *Input1, const NumType *Input2,
                                                        const size_t StartRow, const size_t StopRow, const size_t Cols,
                                                        const size_t TargetSizeOfLine, const size_t Input1SizeOfLine,
                                                        const size_t Input2SizeOfLine) {
    if (StartRow == StopRow) return;

    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = 0; j < ElementsInCacheLine; ++j) {
            for (size_t k = StartRow; k < StopRow; ++k) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (size_t i = ColsRange; i < Cols; ++i) {
        for (size_t k = StartRow; k < StopRow; ++k) {
            Target[i * TargetSizeOfLine + k] = Input1[i * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const size_t StartCol,
                                                        const size_t StopCol, const size_t Rows,
                                                        const size_t TargetSizeOfLine,
                                                        const size_t Input1SizeOfLine,
                                                        const size_t Input2SizeOfLine) {
    if (StartCol == StopCol) return;

    const size_t ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const size_t RowsRange = ((size_t)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i+= ElementsInCacheLine) {
        for (size_t k = StartCol; k < StopCol; ++k) {
            for (size_t j = 0; j < ElementsInCacheLine; ++j) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (size_t i = RowsRange; i < Rows; ++i) {
        for (size_t j = StartCol; j < StopCol; ++j)
            Target[i * TargetSizeOfLine + j] = Input1[i * TargetSizeOfLine + j] + Input2[j * TargetSizeOfLine + i];
    }
}

// ------------------------------------------
// Matrix Transposition Implementation
// ------------------------------------------

template<typename NumType>
void
TransposeMatrixRowStored(NumType *Dst, NumType *Src, const size_t SrcLines, const size_t SrcElementsPerLine,
                         const size_t DstSizeOfLine, const size_t SrcSizeOfLine) {
    for (size_t i = 0; i < SrcLines; ++i) {
        for (size_t j = 0; j < SrcElementsPerLine; ++j) {
            Dst[j * DstSizeOfLine + i] = Src[i * SrcSizeOfLine + j];
        }
    }
}

// ------------------------------------------
// Matrix Dot product Implementation
// ------------------------------------------

template<typename NumType>
NumType DotProduct(NumType *const Src1, NumType *const Src2, const size_t Range) {
    NumType result = NumType();

    for (size_t i = 0; i < Range; i++) {
        result += Src1[i] * Src2[i];
    }

    return result;
}

template<typename NumType>
NumType DPMCore<NumType>::GetResult() {
    NumType Ret = NumType();

    for (size_t i = EndIndex; i < Range; ++i) {
        Ret += Src1[i] * Src2[i];
    }

    for (unsigned i = 0; i < Threads; ++i) {
        Ret += ResultArray[i];
    }

    return Ret;
}

template<typename NumType>
void DotProductMachineChunked<NumType>::StartThread(unsigned int ThreadID) {
    NumType Ret = NumType();
    const NumType* const S1 = DPMCore<NumType>::Src1;
    const NumType* const S2 = DPMCore<NumType>::Src2;
    const size_t LoopRange = (ThreadID + 1) * ElemPerThread;

    DPMCore<NumType>::Counter.arrive_and_wait();
    for (size_t i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
        Ret += S1[i] * S2[i];
    }

    DPMCore<NumType>::WriteCounter.arrive_and_wait();
    DPMCore<NumType>::ResultArray[ThreadID] = Ret;
}


#define SingleOPStraight(offset) TempArray[offset] += S1[i + offset] * S2[i + offset]

template<typename T>
void DotProductMachineComb<T>::StartThread(unsigned int ThreadID) {
    T* TempArray = new T[PerCircle]{ 0 };
    const T* const S1 = DPMCore<T>::Src1 + (ThreadID * PerCircle);
    const T* const S2 = DPMCore<T>::Src2 + (ThreadID * PerCircle);

    const size_t Jump = PerCircle * DPMCore<T>::Threads;

    DPMCore<T>::Counter.arrive_and_wait();

    for (size_t i = 0; i < LoopRange; i += Jump) {
        for (size_t j = 0; j < PerCircle; ++j) {
            SingleOPStraight(j);
        }
    }

    DPMCore<T>::WriteCounter.arrive_and_wait();
    T Result = T();

    for (size_t i = 0; i < PerCircle; ++i) {
        Result += TempArray[i];
    }

    DPMCore<T>::ResultArray[ThreadID] = Result;
    delete[] TempArray;
}

// ------------------------------------------
// Outer Product Implementation
// ------------------------------------------

template<typename NumType>
OPM<NumType>::OPM(const NumType *VectA, const NumType *VectB, NumType *MatC, size_t ASize, size_t BSize,
                  size_t MatCSoL, bool IsHor):
    MatC{ MatC }, MatCSoL{ MatCSoL }
{
    if (IsHor){
        CoefPtr = VectA;
        VectPtr = VectB;
        CoefSize = ASize;
        VectSize = BSize;
    }
    else{
        CoefPtr = VectB;
        VectPtr = VectA;
        CoefSize = BSize;
        VectSize = ASize;
    }
}


template<typename NumType>
void OPM<NumType>::Perform() {
    const size_t Range = (CoefSize / ElementsPerCacheLine) * ElementsPerCacheLine;
    for (size_t i = 0; i < Range; i += ElementsPerCacheLine) {
        for (size_t j = 0; j < VectSize; ++j) {
            MatC[i * MatCSoL + j] = VectPtr[j] * CoefPtr[i];
            MatC[(i + 1) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 1];
            MatC[(i + 2) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 2];
            MatC[(i + 3) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 3];
            MatC[(i + 4) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 4];
            MatC[(i + 5) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 5];
            MatC[(i + 6) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 6];
            MatC[(i + 7) * MatCSoL + j] = VectPtr[j] * CoefPtr[i + 7];
        }
    }

    for(size_t  j = 0; j < VectSize; ++j) {
        for (size_t i = Range; i < CoefSize; ++i) {
            MatC[i * MatCSoL + j] = VectPtr[j] * CoefPtr[i];
        }
    }
}

#ifdef __AVX__
template<>
inline void OPM<double>::Perform()
#define LoadAvx(double_ptr) *(__m256d*)double_ptr
{
    const size_t CoefRange = (CoefSize / ElementsPerCacheLine) * ElementsPerCacheLine;
    const double *CoefPtrIter = CoefPtr;

#pragma omp parallel for
    for (size_t i = 0; i < CoefRange; i += ElementsPerCacheLine) {
        __m256d CoefBuff0 = _mm256_set1_pd(*(CoefPtrIter));
        __m256d CoefBuff1 = _mm256_set1_pd(*(CoefPtrIter + 1));
        __m256d CoefBuff2 = _mm256_set1_pd(*(CoefPtrIter + 2));
        __m256d CoefBuff3 = _mm256_set1_pd(*(CoefPtrIter + 3));
        __m256d CoefBuff4 = _mm256_set1_pd(*(CoefPtrIter + 4));
        __m256d CoefBuff5 = _mm256_set1_pd(*(CoefPtrIter + 5));
        __m256d CoefBuff6 = _mm256_set1_pd(*(CoefPtrIter + 6));
        __m256d CoefBuff7 = _mm256_set1_pd(*(CoefPtrIter + 7));
        CoefPtrIter += ElementsPerCacheLine;

        const double *VectPtrIter = VectPtr;
        for (size_t j = 0; j < VectSize; j += ElementsPerCacheLine) {
            __m256d VectA0 = _mm256_load_pd(VectPtrIter);
            __m256d VectA1 = _mm256_load_pd(VectPtrIter + DOUBLE_VECTOR_LENGTH);
            VectPtrIter += ElementsPerCacheLine;

            double *TargetFirstPtr0 = MatC + i * MatCSoL + j;
            double *TargetSecondPtr0 = MatC + i * MatCSoL + j + DOUBLE_VECTOR_LENGTH;
            double *TargetFirstPtr1 = MatC + (i + 2) * MatCSoL + j;
            double *TargetSecondPtr1 = MatC + (i + 2) * MatCSoL + j + DOUBLE_VECTOR_LENGTH;
            double *TargetFirstPtr2 = MatC + (i + 4) * MatCSoL + j;
            double *TargetSecondPtr2 = MatC + (i + 4) * MatCSoL + j + DOUBLE_VECTOR_LENGTH;
            double *TargetFirstPtr3 = MatC + (i + 6) * MatCSoL + j;
            double *TargetSecondPtr3 = MatC + (i + 6) * MatCSoL + j + DOUBLE_VECTOR_LENGTH;
            LoadAvx(TargetFirstPtr0) = _mm256_mul_pd(VectA0, CoefBuff0);
            LoadAvx(TargetSecondPtr0) = _mm256_mul_pd(VectA1, CoefBuff0);
            LoadAvx(TargetFirstPtr1) = _mm256_mul_pd(VectA0, CoefBuff2);
            LoadAvx(TargetSecondPtr1) = _mm256_mul_pd(VectA1, CoefBuff2);
            LoadAvx(TargetFirstPtr2) = _mm256_mul_pd(VectA0, CoefBuff4);
            LoadAvx(TargetSecondPtr2) = _mm256_mul_pd(VectA1, CoefBuff4);
            LoadAvx(TargetFirstPtr3) = _mm256_mul_pd(VectA0, CoefBuff6);
            LoadAvx(TargetSecondPtr3) = _mm256_mul_pd(VectA1, CoefBuff6);
            TargetSecondPtr0 += MatCSoL;
            TargetFirstPtr0 += MatCSoL;
            TargetFirstPtr1 += MatCSoL;
            TargetSecondPtr1 += MatCSoL;
            TargetFirstPtr2 += MatCSoL;
            TargetSecondPtr2 += MatCSoL;
            TargetFirstPtr3 += MatCSoL;
            TargetSecondPtr3 += MatCSoL;
            LoadAvx(TargetFirstPtr0) = _mm256_mul_pd(VectA0, CoefBuff1);
            LoadAvx(TargetSecondPtr0) = _mm256_mul_pd(VectA1, CoefBuff1);
            LoadAvx(TargetFirstPtr1) = _mm256_mul_pd(VectA0, CoefBuff3);
            LoadAvx(TargetSecondPtr1) = _mm256_mul_pd(VectA1, CoefBuff3);
            LoadAvx(TargetFirstPtr2) = _mm256_mul_pd(VectA0, CoefBuff5);
            LoadAvx(TargetSecondPtr2) = _mm256_mul_pd(VectA1, CoefBuff5);
            LoadAvx(TargetFirstPtr3) = _mm256_mul_pd(VectA0, CoefBuff7);
            LoadAvx(TargetSecondPtr3) = _mm256_mul_pd(VectA1, CoefBuff7);
        }
    }

    const size_t CleaningRange = CoefSize - CoefRange;
    __m256d Buffers[ElementsPerCacheLine];
    for (size_t i = 0; i < CleaningRange; i++) {
        Buffers[i] = _mm256_set1_pd(CoefPtr[CoefRange + i]);
    }

    for (size_t j = 0; j < VectSize; j += ElementsPerCacheLine) {
        __m256d VectFirst = _mm256_load_pd(VectPtr + j);
        __m256d VectSecond = _mm256_load_pd(VectPtr + j + DOUBLE_VECTOR_LENGTH);

        for (size_t i = 0; i < CleaningRange; i++)
#define CleaningTargetUpper MatC + (i + CoefRange) * MatCSoL + j
#define CleaningTargetLower MatC + (i + CoefRange) * MatCSoL + j + DOUBLE_VECTOR_LENGTH
        {
            _mm256_store_pd(CleaningTargetUpper, _mm256_mul_pd(VectFirst, Buffers[i]));
            _mm256_store_pd(CleaningTargetLower, _mm256_mul_pd(VectSecond, Buffers[i]));
        }
    }
}

#endif

// ------------------------------------------
// Matrix and Vector Multiplication Implementation
// ------------------------------------------

template<typename NumType>
VMM<NumType>::VMM(const NumType *MatA, const NumType *VectB, NumType *VectC, size_t MatARows, size_t MatACols,
                  size_t MatASoL, bool IsHor) :
    MatA{ MatA }, VectB{ VectB }, VectC { VectC }, MatARows{ MatARows }, MatACols { MatACols },
    MatASoL{ MatASoL }, IsMatHor { IsHor }
{

}

template<typename NumType>
void VMM<NumType>::PerformCMV() {
    const size_t Range = (MatARows / 4) * 4;

    for(size_t i = 0; i < Range; i+=4){
        NumType acc0 = 0;
        NumType acc1 = 0;
        NumType acc2 = 0;
        NumType acc3 = 0;
        for(size_t j = 0; j < MatACols; ++j){
            acc0 += MatA[j * MatASoL + i] * VectB[j];
            acc1 += MatA[j * MatASoL + i + 1] * VectB[j];
            acc2 += MatA[j * MatASoL + i + 2] * VectB[j];
            acc3 += MatA[j * MatASoL + i + 3] * VectB[j];
        }
        VectC[i] = acc0;
        VectC[i + 1] = acc1;
        VectC[i + 2] = acc2;
        VectC[i + 3] = acc3;
    }

    if (Range == MatARows) return;
    // Otherwise, perform cleaning of rows, which are not packable in fours

    NumType Accumulators[4] = { 0 };
    const size_t CleaningRange = MatARows - Range;
    for (size_t j = 0; j < MatACols; ++j){
        for(size_t i = 0; i < CleaningRange; ++i){
            Accumulators[i] += MatA[j * MatASoL + Range + i] * VectB[j];
        }
    }

    for(size_t i = 0; i < CleaningRange; ++i){
        VectC[Range + i] = Accumulators[i];
    }
}
template<typename NumType>
void VMM<NumType>::PerformRMV() {
    for(size_t i = 0; i < MatARows; i+=4){
        NumType acc0 = 0;
        NumType acc1 = 0;
        NumType acc2 = 0;
        NumType acc3 = 0;
        for(size_t j = 0; j < MatACols; ++j){
            acc0 += MatA[i * MatASoL + j] * VectB[j];
            acc1 += MatA[(i + 1) * MatASoL + j] * VectB[j];
            acc2 += MatA[(i + 2) * MatASoL + j] * VectB[j];
            acc3 += MatA[(i + 3) * MatASoL + j] * VectB[j];
        }
        VectC[i] = acc0;
        VectC[i + 1] = acc1;
        VectC[i + 2] = acc2;
        VectC[i + 3] = acc3;
    }

    // Cleaning is not necessary, if there is no memory optimization, due to alignment
}

template<typename NumType>
void VMM<NumType>::PerformCVM() {

}

#endif // PARALLELNUM_NUMERICAL_CORE_H_