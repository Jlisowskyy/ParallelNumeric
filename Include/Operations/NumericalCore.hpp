
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
void MatrixSumHelperAlignedArrays(NumType*Target, const NumType* Input1, const NumType* Input2, unsigned long Elements);
// Function used to Sum matrices

#ifdef __AVX__
//defined(__AVX__) && defined(__FMA__)
template<>
void MatrixSumHelperAlignedArrays(double* Target,const double* const Input1, const double* const Input2, const unsigned long Elements);

template<>
void MatrixSumHelperAlignedArrays(float* Target, const float* const Input1, const float* const Input2, const unsigned long Elements);

#endif // __AVX__

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  unsigned StartCol, unsigned StopCol, unsigned Rows,
                                                  unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  unsigned StartRow, unsigned StopRow, unsigned Cols,
                                                  unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  unsigned StartCol, unsigned StopCol, unsigned Rows,
                                                  unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                  unsigned StartRow, unsigned StopRow, unsigned Cols,
                                                  unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        unsigned StartCol, unsigned StopCol, unsigned Rows,
                                                        unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        unsigned StartRow, unsigned StopRow, unsigned Cols,
                                                        unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        unsigned StartRow, unsigned StopRow, unsigned Cols,
                                                        unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame(NumType* Target, const NumType* Input1, const NumType* Input2,
                                                        unsigned StartCol, unsigned StopCol, unsigned Rows,
                                                        unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine);

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
void TransposeMatrixRowStored(NumType* Dst, NumType* Src, unsigned SrcLines, unsigned SrcElementsPerLine,
                              unsigned DstSizeOfLine, unsigned SrcSizeOfLine);

// ------------------------------------------
// Dot product code
// ------------------------------------------

template<typename NumType>
NumType DotProduct(NumType* Src1, NumType* Src2, unsigned long Range);

#ifdef __AVX__

template<>
double DotProduct(double * Src1, double * Src2, unsigned long Range);

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
	const unsigned long Range;
	const unsigned long EndIndex;
	NumType ResultArray[MaxCPUThreads] = {NumType() };
	std::latch Counter;
	std::latch WriteCounter;
public:
	DPMCore(const NumType* const Src1, const NumType* const Src2, unsigned Threads, unsigned long Range, unsigned long EndIndex) :
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
	const unsigned long ElemPerThread;
public:
	DotProductMachineChunked(const NumType* const Src1, const NumType* const Src2, unsigned Threads, unsigned long Range) :
		ElemPerThread{ Range / (unsigned long) Threads }, DPMCore<NumType>(Src1, Src2, Threads, Range, (Range / (unsigned long)Threads) * (unsigned long)Threads)
	{}

	void StartThread(unsigned ThreadID);
};

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* Src1, const double* Src2, unsigned Threads, unsigned long Range);

template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* Src1, const float* Src2, unsigned Threads, unsigned long Range);

template<>
void DotProductMachineChunked<double>::StartThread(unsigned ThreadID);

template<>
void DotProductMachineChunked<float>::StartThread(unsigned ThreadID);

#endif

template<typename NumType>
class DotProductMachineComb: public DPMCore<NumType> {
	const unsigned long LoopRange;
	const unsigned long PerCircle = CACHE_LINE / sizeof(NumType);
public:
	DotProductMachineComb(const NumType* const Src1, const NumType* const Src2, unsigned Threads, unsigned long Range) :
		LoopRange{ Range }, DPMCore<NumType>(Src1, Src2, Threads, Range, (Range / (CACHE_LINE / sizeof(NumType))) * (CACHE_LINE / sizeof(NumType)))
	{}

	void StartThread(unsigned ThreadID);
};

#if defined(__AVX__) && defined(__FMA__)

template<>
DotProductMachineComb<double>::DotProductMachineComb(const double* Src1, const double* Src2, unsigned Threads, unsigned long Range);

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* Src1, const float* Src2, unsigned Threads, unsigned long Range);

template<>
void DotProductMachineComb<double>::StartThread(unsigned ThreadID);

template<>
void DotProductMachineComb<float>::StartThread(unsigned ThreadID);

#endif

// ------------------------------------------
// Outer Product
// ------------------------------------------

template<typename NumType>
void OuterProductCol(NumType* Dst, const NumType* const Src1, const NumType* const Src2, std::pair<unsigned, unsigned> Dim) {
	for (unsigned i = 0; i < Dim.second; ++i) {
		for (unsigned j = 0; j < Dim.first; ++j) {
			Dst[i * Dim.first + j] = Src1[j] * Src2[i];
		}
	}
}

// ------------------------------------------
// Matrix Sum Implementation
// ------------------------------------------

template<typename T>
void MatrixSumHelperAlignedArrays(T *const Target, const T *const Input1, const T *const Input2,
                                  const unsigned long Elements) {
    for (unsigned long i = 0; i < Elements; ++i)
        Target[i] = Input1[i] + Input2[i];
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                                  const unsigned int StartCol, const unsigned int StopCol,
                                                  const unsigned int Rows, const unsigned int TargetSizeOfLine,
                                                  const unsigned int Input1SizeOfLine,
                                                  const unsigned int Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned RowsRange = ((unsigned)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (unsigned j = StartCol; j < StopCol; j += ElementsInCacheLine) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
                for (unsigned z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(i + z) * TargetSizeOfLine + j + k] = Input1[(i + z) * Input1SizeOfLine + j + k] + Input2[(j + k) * Input2SizeOfLine + i + z];
                }
            }
        }
    }

    for (unsigned i = StartCol; i < StopCol; ++i) {
        for (unsigned j = RowsRange; j < Rows; ++j) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
                Target[j * TargetSizeOfLine + i + k] = Input1[j * Input1SizeOfLine + i + k] + Input2[(i + k) * Input2SizeOfLine + j];
            }
        }
    }
}

template<typename NumType>
void
MatrixSumHelperNotAlignedArrays_RC_DivByRows(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                             const unsigned int StartRow, const unsigned int StopRow,
                                             const unsigned int Cols, const unsigned int TargetSizeOfLine,
                                             const unsigned int Input1SizeOfLine, const unsigned int Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned ColsRange = ((unsigned)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned j = StartRow; j < StopRow; j += ElementsInCacheLine) {
        for (unsigned i = 0; i < ColsRange; i += ElementsInCacheLine) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
                for (unsigned z = 0; z < ElementsInCacheLine; ++z) {
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
                                             const unsigned int StartCol, const unsigned int StopCol,
                                             const unsigned int Rows, const unsigned int TargetSizeOfLine,
                                             const unsigned int Input1SizeOfLine, const unsigned int Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned RowsRange = ((unsigned) (Rows / ElementsInCacheLine)) * ElementsInCacheLine;
    for (unsigned j = StartCol; j < StopCol; j += ElementsInCacheLine) {
        for (unsigned i = 0; i < RowsRange; i += ElementsInCacheLine) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
                for (unsigned z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(j + k) * TargetSizeOfLine + i + z] = Input1[(j + k) * Input1SizeOfLine + i + z] + Input2[(i + z) * Input2SizeOfLine + j + k];
                }
            }
        }
    }

    for (unsigned j = StartCol; j < StopCol; ++j) {
        for (unsigned z = RowsRange; z < Rows; ++z) {
            Target[j * TargetSizeOfLine + z] = Input1[j * Input1SizeOfLine + z] + Input2[z * Input2SizeOfLine + j];
        }
    }
}

template<typename NumType>
void
MatrixSumHelperNotAlignedArrays_CR_DivByRows(NumType *Target, const NumType *const Input1, const NumType *const Input2,
                                             const unsigned int StartRow, const unsigned int StopRow,
                                             const unsigned int Cols, const unsigned int TargetSizeOfLine,
                                             const unsigned int Input1SizeOfLine, const unsigned int Input2SizeOfLine)
// Function assumes that passed Start and StopCol ale divisible by NumType corresponding length of cache line;
// otherwise, behavior is undefined.
// Also, all data pointers should be aligned to cache lines;
// otherwise, the operation may be much slower.
{
    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned ColsRange = ((unsigned)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (unsigned j = StartRow; j < StopRow; j += ElementsInCacheLine) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
                for (unsigned z = 0; z < ElementsInCacheLine; ++z) {
                    Target[(i + k) * TargetSizeOfLine + j + z] = Input1[(i + k) * Input1SizeOfLine + j + z] + Input2[(j + z) * Input2SizeOfLine + i + k];
                }
            }
        }
    }

    for (unsigned i = StartRow; i < StopRow; i += ElementsInCacheLine) {
        for (unsigned z = ColsRange; z < Cols; ++z) {
            for (unsigned k = 0; k < ElementsInCacheLine; ++k)
                Target[z * TargetSizeOfLine + i + k] = Input1[z * Input1SizeOfLine + i + k] + Input2[(i + k) * Input2SizeOfLine + z];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const unsigned int StartCol,
                                                        const unsigned int StopCol, const unsigned int Rows,
                                                        const unsigned int TargetSizeOfLine,
                                                        const unsigned int Input1SizeOfLine,
                                                        const unsigned int Input2SizeOfLine) {
    if (StartCol == StopCol) return;

    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned RowsRange = ((unsigned)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (unsigned k = StartCol; k < StopCol; ++k) {
            for (unsigned j = 0; j < ElementsInCacheLine; ++j) {
                Target[k * TargetSizeOfLine + i + j] = Input1[k * Input1SizeOfLine + i + j] + Input2[(i + j) * Input2SizeOfLine + k];
            }
        }
    }

    for (unsigned i = RowsRange; i < Rows; ++i) {
        for (unsigned k = StartCol; k < StopCol; ++k) {
            Target[k * TargetSizeOfLine + i] = Input1[k * Input1SizeOfLine + i] + Input2[i * Input2SizeOfLine + k];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const unsigned int StartRow,
                                                        const unsigned int StopRow, const unsigned int Cols,
                                                        const unsigned int TargetSizeOfLine,
                                                        const unsigned int Input1SizeOfLine,
                                                        const unsigned int Input2SizeOfLine) {
    if (StartRow == StopRow) return;

    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned ColsRange = ((unsigned)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < ElementsInCacheLine; ++j) {
            for (unsigned k = StartRow; k < StopRow; ++k) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (unsigned i = ColsRange; i < Cols; ++i) {
        for (unsigned k = StartRow; k < StopRow; ++k) {
            Target[i * TargetSizeOfLine + k] = Input1[i * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame(NumType *Target, const NumType *Input1, const NumType *Input2,
                                                        unsigned int StartRow, unsigned int StopRow, unsigned int Cols,
                                                        unsigned int TargetSizeOfLine, unsigned int Input1SizeOfLine,
                                                        unsigned int Input2SizeOfLine) {
    if (StartRow == StopRow) return;

    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned ColsRange = ((unsigned)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (unsigned j = 0; j < ElementsInCacheLine; ++j) {
            for (unsigned k = StartRow; k < StopRow; ++k) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (unsigned i = ColsRange; i < Cols; ++i) {
        for (unsigned k = StartRow; k < StopRow; ++k) {
            Target[i * TargetSizeOfLine + k] = Input1[i * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i];
        }
    }
}

template<typename NumType>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame(NumType *Target, const NumType *const Input1,
                                                        const NumType *const Input2, const unsigned int StartCol,
                                                        const unsigned int StopCol, const unsigned int Rows,
                                                        const unsigned int TargetSizeOfLine,
                                                        const unsigned int Input1SizeOfLine,
                                                        const unsigned int Input2SizeOfLine) {
    if (StartCol == StopCol) return;

    const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(NumType);
    const unsigned RowsRange = ((unsigned)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (unsigned i = 0; i < RowsRange; i+= ElementsInCacheLine) {
        for (unsigned k = StartCol; k < StopCol; ++k) {
            for (unsigned j = 0; j < ElementsInCacheLine; ++j) {
                Target[(i + j) * TargetSizeOfLine + k] = Input1[(i + j) * Input1SizeOfLine + k] + Input2[k * Input2SizeOfLine + i + j];
            }
        }
    }

    for (unsigned i = RowsRange; i < Rows; ++i) {
        for (unsigned j = StartCol; j < StopCol; ++j)
            Target[i * TargetSizeOfLine + j] = Input1[i * TargetSizeOfLine + j] + Input2[j * TargetSizeOfLine + i];
    }
}

// ------------------------------------------
// Matrix Transposition Implementation
// ------------------------------------------

template<typename NumType>
void
TransposeMatrixRowStored(NumType *Dst, NumType *Src, const unsigned int SrcLines, const unsigned int SrcElementsPerLine,
                         const unsigned int DstSizeOfLine, const unsigned int SrcSizeOfLine) {
    for (unsigned i = 0; i < SrcLines; ++i) {
        for (unsigned j = 0; j < SrcElementsPerLine; ++j) {
            Dst[j * DstSizeOfLine + i] = Src[i * SrcSizeOfLine + j];
        }
    }
}

// ------------------------------------------
// Matrix Dot product Implementation
// ------------------------------------------

template<typename NumType>
NumType DotProduct(NumType *const Src1, NumType *const Src2, const unsigned long Range) {
    NumType result = NumType();

    for (unsigned long i = 0; i < Range; i++) {
        result += Src1[i] * Src2[i];
    }

    return result;
}

template<typename NumType>
NumType DPMCore<NumType>::GetResult() {
    NumType Ret = NumType();

    for (unsigned long i = EndIndex; i < Range; ++i) {
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
    const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;

    DPMCore<NumType>::Counter.arrive_and_wait();
    for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
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

    const unsigned long Jump = PerCircle * DPMCore<T>::Threads;

    DPMCore<T>::Counter.arrive_and_wait();

    for (unsigned long i = 0; i < LoopRange; i += Jump) {
        for (unsigned long j = 0; j < PerCircle; ++j) {
            SingleOPStraight(j);
        }
    }

    DPMCore<T>::WriteCounter.arrive_and_wait();
    T Result = T();

    for (int i = 0; i < PerCircle; ++i) {
        Result += TempArray[i];
    }

    DPMCore<T>::ResultArray[ThreadID] = Result;
    delete[] TempArray;
}

// ------------------------------------------
// Outer Product
// ------------------------------------------

//#define pt (const __m256d* const)
//#define LOB(offset) VectDst[offset][0] = _mm256_mul_pd(*(pt(Src1 + j)), Mult[offset]); VectDst[offset][1] = _mm256_mul_pd(*(pt(Src1 + j + 4)), Mult[offset]);
//#define LoadDst(offset) (__m256d*) (Dst + (i + offset) * Dim.first);
//
//template<>
//void OuterProductCol(double* Dst, const double* const Src1, const double* const Src2, std::pair<unsigned, unsigned> Dim) {
//    //const unsigned long BlockSize = 8;
//    //const unsigned long BlockedRangeHorizontal =
//    //	Dim.second >= BlockSize ? Dim.second - BlockSize : 0; // CHECK
//    //const unsigned long BlockedRangeVertical =
//    //	Dim.first >= BlockSize ? Dim.first - BlockSize : 0; // CHECK
//
//    //for (unsigned long i = 0; i < BlockedRangeHorizontal; i += BlockSize) {
//    //	for (unsigned long j = 0; j < BlockedRangeVertical; j += BlockSize) {
//    //		for (unsigned long k = i; k < i + BlockSize; ++k) {
//    //			for (unsigned long z = j; z < j + BlockSize; ++z) {
//    //				Dst[k * Dim.second + z] = Src1[k] * Src2[z];
//    //			}
//    //		}
//    //	}
//    //}
//
//    const unsigned long BlockSize = 8;
//    const unsigned long BlockedRangeHorizontal =
//            Dim.second >= BlockSize ? Dim.second - BlockSize : 0; // CHECK
//    const unsigned long BlockedRangeVertical =
//            Dim.first >= BlockSize ? Dim.first - BlockSize : 0; // CHECK
//
//    const auto S1 = (const __m256d* const) (Src1);
//    auto VectDst = (__m256d**) _aligned_malloc(sizeof(__m256d*) * BlockSize, ALIGN);
//    auto Mult = (__m256d*) _aligned_malloc(sizeof(__m256d) * BlockSize, ALIGN);
//
//    if (!VectDst || !Mult) {
//        std::cout << "alloc err\n";
//        exit(0);
//    }
//
//    for (unsigned long i = 0; i < BlockedRangeHorizontal; i += BlockSize) {
//
//        for (int z = 0; z < BlockSize; ++z) {
//            VectDst[z] = LoadDst(z);
//            Mult[z] = _mm256_set1_pd(Src2[i + z]);
//        }
//
//        for (unsigned long j = 0; j < BlockedRangeVertical; j += BlockSize) {
//            for (int z = 0; z < BlockSize; ++z) {
//                LOB(z);
//            }
//        }
//
//        _mm256_add_epi64(((__m256i*)VectDst)[0], _mm256_set1_epi64x(2));
//        _mm256_add_epi64(((__m256i*)VectDst)[1], _mm256_set1_epi64x(2));
//    }
//
//    _aligned_free(Mult);
//    _aligned_free(VectDst);
//}

#endif // PARALLELNUM_HELPERS_H_