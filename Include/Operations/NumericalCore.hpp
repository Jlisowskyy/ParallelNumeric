
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

// ------------------------------------
// Thread Number management
// ------------------------------------

template<unsigned ThreadCap = MaxCPUThreads>
inline unsigned LogarithmicThreads(unsigned long long);

template<unsigned ThreadCap = MaxCPUThreads>
inline unsigned LinearThreads(unsigned long long);

static struct MatrixMultThreadsDecider
    // TODO: Upgrade performance
{
       static constexpr float RefusalThreshold = 0.75f;
       static const int MaximalIterations = 4;
       static const long long unsigned StartingThreshold = 32768;

       template<unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LogarithmicThreads<ThreadCap>>
       inline bool FindOptimalThreadNumber(std::pair<unsigned, unsigned>& RetVal, unsigned Blocks, unsigned long long OperationCount);
} MMThreads;

class MatrixMultInterface{
public:

    virtual void ProcessBlock(unsigned, unsigned) = 0;
    virtual void ProcessFrame() = 0;
    virtual ~MatrixMultInterface() = default;
};

class MatrixMultThreadExecutionUnit{
    MatrixMultInterface* Machine;
    const unsigned ThreadCount;
    unsigned BlocksPerThread, Blocks;
    std::latch Synchronizer;
    std::mutex m;
    bool FrameDone = false;

    // TODO: find something better than framedone XD
public:
    MatrixMultThreadExecutionUnit(MatrixMultInterface* Machine, unsigned ThreadCount, unsigned BlocksPerThread, unsigned Blocks) :
    Machine{Machine}, ThreadCount {ThreadCount}, BlocksPerThread{BlocksPerThread}, Blocks{Blocks}, Synchronizer(ThreadCount){}

    void StartExecution();

private:
    void MatrixMultThread(unsigned StartBlock, unsigned BorderBlock);
};


// ------------------------------------------
// Sum of matrices functions

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

// ------------------------------------
// Matrix Multiplication

template<typename NumType>
class CCTarHor_MultMachine: public MatrixMultInterface {
public:
    static constexpr unsigned BlockSize = MATRIX_MULT_BLOCK_COEF * (CACHE_LINE / sizeof(NumType));
private:
	const unsigned Src1Rows;
	const unsigned Src1Cols;
	const unsigned Src2Rows;
	const unsigned Src2Cols;
	const unsigned TargetSizeOfLine;
	const unsigned Src1SizeOfLine;
	const unsigned Src2SizeOfLine;
	NumType* const Target;
	NumType* const Src1;
	NumType* const Src2;
    unsigned VectorBlocksRange;
    unsigned BlocksPerVectorRange;
    unsigned BlocksPerBaseVectorRange;
    void (CCTarHor_MultMachine::*MainFunc)(unsigned, unsigned);
    void (CCTarHor_MultMachine::*FrameFunc)();

public:
    CCTarHor_MultMachine(unsigned Src1Rows, unsigned Src1Cols, unsigned Src2Rows, unsigned Src2Cols,
                         unsigned TargetSizeOfLine, unsigned Src1SizeOfLine, unsigned Src2SizeOfLine,
                         NumType* Target, NumType* Src1, NumType* Src2);

    inline void ProcessAllBlocks(){
        (this->*MainFunc)(0, VectorBlocksRange);
    }

    inline void ProcessBlock(unsigned StartingBlock, unsigned BorderBlock) final {
        (this->*MainFunc)(StartingBlock * BlockSize, BorderBlock * BlockSize);
    }

    inline void ProcessFrame() final {
        if (VectorBlocksRange != Src2Cols) (this->*FrameFunc)();
    }

#define BBScaledVectorCoefPacked(offset) Src1[(j + jj) * Src1SizeOfLine + k + kk + offset] * VectorScalar
	// Scaled vector coef, used to get multiple scaled coefs with single val at once(4)
#define BBSaveAccumulatedCoefsToTarget(offset) Target[(i + ii) * TargetSizeOfLine + k + kk]
	// After accumulating desired number of vectors coef accumulator variable is added to
	// proper target storage

#define NBScaledVectorCoefPacked(offset) (acc0 += Src1[j * Src1SizeOfLine + k + kk + offset] * VectorScalar)

private:
	void EEBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);
	void ENBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);
	void NEBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);
	void NNBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);
	void EEFrame();
	void ENFrame();
	void NEFrame();
	void NNFrame();
};

#if defined(__AVX__) && defined(__FMA__)

template<>
void CCTarHor_MultMachine<double>::EEBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);

template<>
void CCTarHor_MultMachine<double>::ENBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);

template<>
void CCTarHor_MultMachine<double>::NEBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);

template<>
void CCTarHor_MultMachine<double>::NNBlocks(unsigned VectorStartingBlock, unsigned VectorBlocksBorder);

#endif

// -------------------------------------
// Matrix transposition solutions

// Naive solution
template<typename NumType>
void TransposeMatrixRowStored(NumType* Dst, NumType* Src, unsigned SrcLines, unsigned SrcElementsPerLine,
                              unsigned DstSizeOfLine, unsigned SrcSizeOfLine);

// ---------------------------------
// Dot product code

template<typename NumType>
NumType DotProduct(NumType* Src1, NumType* Src2, unsigned long Range);

#ifdef __AVX__

template<>
double DotProduct(double * Src1, double * Src2, unsigned long Range);

#endif // __AVX__

// ---------------------------------
// Solution 1

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

// ---------------------------------
// Solution 2

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

// ---------------------------------
// Outer Product

template<typename NumType>
void OuterProductCol(NumType* Dst, const NumType* const Src1, const NumType* const Src2, std::pair<unsigned, unsigned> Dim) {
	for (unsigned i = 0; i < Dim.second; ++i) {
		for (unsigned j = 0; j < Dim.first; ++j) {
			Dst[i * Dim.first + j] = Src1[j] * Src2[i];
		}
	}
}

//----------------------------------
// Thread management Implementation
//----------------------------------

template<unsigned ThreadCap>
unsigned LogarithmicThreads(const unsigned long long int Elements) {
    auto Ret = (unsigned)(log2((double) (Elements / ThreadedStartingThreshold) )) + 1u;
    return std::min(ThreadCap, Ret);
}

template<unsigned ThreadCap>
unsigned LinearThreads(const unsigned long long int Elements) {
    auto Ret = (unsigned)(Elements / ThreadedStartingThreshold);
    return std::min(ThreadCap, Ret);
}

template<unsigned ThreadCap, unsigned (*Decider)(unsigned long long)>
bool MatrixMultThreadsDecider::FindOptimalThreadNumber(std::pair<unsigned, unsigned>& RetVal, unsigned Blocks, unsigned long long OperationCount)
    // TOTAL MESS
{
    unsigned DesiredThreadAmount = Decider(OperationCount);
    unsigned BlocksPerThread = Blocks / DesiredThreadAmount;

    RetVal.first = DesiredThreadAmount;
    RetVal.second = BlocksPerThread;

    if (BlocksPerThread == 0 &&
        (RetVal.first = Blocks % DesiredThreadAmount) >= (unsigned)(RefusalThreshold * (float)DesiredThreadAmount))
        // Shortens the number of threads to equally divide blocks between them in maximal cost
        // dictated by RefusalThreshold. Designed to favor higher hierarchy algorithms.
    {
        RetVal.second = 1;
        return true;
    }
    else if (BlocksPerThread == 0){
        return false;
    }
    else{
        unsigned NotThreadedBlocks = Blocks - DesiredThreadAmount * BlocksPerThread;

        if (NotThreadedBlocks <= BlocksPerThread){;
            return true;
        }
        else{
            int Range = MaximalIterations / 2;
            for(int i = std::max(0,(int) DesiredThreadAmount - Range); i <= (int)std::min(DesiredThreadAmount + Range, ThreadCap); ++i){
                unsigned NewThreadAmount = i;
                BlocksPerThread = Blocks / NewThreadAmount;
                NotThreadedBlocks = Blocks - NewThreadAmount * BlocksPerThread;

                if (NotThreadedBlocks <= BlocksPerThread){
                    RetVal.first = NewThreadAmount;
                    RetVal.second = BlocksPerThread;

                    return true;
                }
            }

            return false;
        }
    }
}

//----------------------------------
// Matrix Sum Implementation
//----------------------------------

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

// -----------------------------------------
// Matrix mult implementation
// -----------------------------------------

template<typename NumType>
CCTarHor_MultMachine<NumType>::CCTarHor_MultMachine(const unsigned int Src1Rows, const unsigned int Src1Cols,
                                                    const unsigned int Src2Rows, const unsigned int Src2Cols,
                                                    const unsigned int TargetSizeOfLine,
                                                    const unsigned int Src1SizeOfLine,
                                                    const unsigned int Src2SizeOfLine, NumType *Target,
                                                    NumType *const Src1, NumType *const Src2) :
        Src1Rows{ Src1Rows }, Src1Cols{ Src1Cols }, Src2Rows{ Src2Rows }, Src2Cols{ Src2Cols },
        TargetSizeOfLine{ TargetSizeOfLine }, Src1SizeOfLine{ Src1SizeOfLine }, Src2SizeOfLine{ Src2SizeOfLine },
        Target{ Target }, Src1{ Src1 }, Src2{ Src2 }
// Decides which variant is most optimal for passed matrices
// TODO: verify
{
    VectorBlocksRange = (Src2Cols / BlockSize) * BlockSize;
    BlocksPerVectorRange = (Src2Rows / BlockSize) * BlockSize;
    BlocksPerBaseVectorRange = (Src1Rows / BlockSize) * BlockSize;

    if (BlocksPerBaseVectorRange == Src1Rows) {
        if (BlocksPerVectorRange == Src2Rows) {
            MainFunc = &CCTarHor_MultMachine::EEBlocks;
            FrameFunc = &CCTarHor_MultMachine::EEFrame;
        }
        else {
            MainFunc = &CCTarHor_MultMachine::ENBlocks;
            FrameFunc = &CCTarHor_MultMachine::ENFrame;
        }
    }
    else {
        if (BlocksPerVectorRange == Src2Rows) {
            MainFunc = &CCTarHor_MultMachine::NEBlocks;
            FrameFunc = &CCTarHor_MultMachine::NEFrame;
        }
        else {
            MainFunc = &CCTarHor_MultMachine::NNBlocks;
            FrameFunc = &CCTarHor_MultMachine::NNFrame;
        }
    }

}

template<typename NumType> void
CCTarHor_MultMachine<NumType>::EEBlocks(const unsigned int VectorStartingBlock, const unsigned int VectorBlocksBorder) {
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += BlockSize) {
        for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
            for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
                for (unsigned ii = 0; ii < BlockSize; ++ii) {
                    for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                        NumType acc0 = 0;
                        NumType acc1 = 0;
                        NumType acc2 = 0;
                        NumType acc3 = 0;

                        for (unsigned jj = 0; jj < BlockSize; ++jj) {
                            NumType VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j + jj];

                            acc0 += BBScaledVectorCoefPacked(0);
                            acc1 += BBScaledVectorCoefPacked(1);
                            acc2 += BBScaledVectorCoefPacked(2);
                            acc3 += BBScaledVectorCoefPacked(3);
                        }


                        BBSaveAccumulatedCoefsToTarget(0) += acc0;
                        BBSaveAccumulatedCoefsToTarget(1) += acc1;
                        BBSaveAccumulatedCoefsToTarget(2) += acc2;
                        BBSaveAccumulatedCoefsToTarget(3) += acc3;

                    }
                }
            }
        }
    }
}

template<typename NumType> void
CCTarHor_MultMachine<NumType>::ENBlocks(const unsigned int VectorStartingBlock, const unsigned int VectorBlocksBorder) {
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += BlockSize) {
        for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
            for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
                for (unsigned ii = 0; ii < BlockSize; ++ii) {
                    for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                        NumType acc0 = 0;
                        NumType acc1 = 0;
                        NumType acc2 = 0;
                        NumType acc3 = 0;

                        for (unsigned jj = 0; jj < BlockSize; ++jj) {
                            NumType VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j + jj];

                            acc0 += BBScaledVectorCoefPacked(0);
                            acc1 += BBScaledVectorCoefPacked(1);
                            acc2 += BBScaledVectorCoefPacked(2);
                            acc3 += BBScaledVectorCoefPacked(3);
                        }

                        BBSaveAccumulatedCoefsToTarget(0) += acc0;
                        BBSaveAccumulatedCoefsToTarget(1) += acc1;
                        BBSaveAccumulatedCoefsToTarget(2) += acc2;
                        BBSaveAccumulatedCoefsToTarget(3) += acc3;

                    }
                }
            }
        }

        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize)
            // Doing last run on the not Blocked Base Vectors
        {
            for (unsigned ii = 0; ii < BlockSize; ++ii) {

                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned j = BlocksPerVectorRange; j < Src1Cols; ++j) {
                        NumType VectorScalar = Src2[(i + ii) * Src2SizeOfLine + j];

                        acc0 += NBScaledVectorCoefPacked(0);
                        acc1 += NBScaledVectorCoefPacked(1);
                        acc2 += NBScaledVectorCoefPacked(2);
                        acc3 += NBScaledVectorCoefPacked(3);
                    }


                    BBSaveAccumulatedCoefsToTarget(0) += acc0;
                    BBSaveAccumulatedCoefsToTarget(1) += acc1;
                    BBSaveAccumulatedCoefsToTarget(2) += acc2;
                    BBSaveAccumulatedCoefsToTarget(3) += acc3;

                }
            }
        }
    }
}
template<typename NumType>
void
CCTarHor_MultMachine<NumType>::NEBlocks(const unsigned int VectorStartingBlock, const unsigned int VectorBlocksBorder) {
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += BlockSize) {
        for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
            for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
                for (unsigned ii = 0; ii < BlockSize; ++ii) {
                    for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                        NumType acc0 = 0;
                        NumType acc1 = 0;
                        NumType acc2 = 0;
                        NumType acc3 = 0;

                        for (unsigned jj = 0; jj < BlockSize; ++jj) {
                            NumType Val = Src2[(i + ii) * Src2SizeOfLine + j + jj];

                            acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                            acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                            acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                            acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                        }


                        Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;

                    }
                }
            }

            // Not guaranteed to be divisible by four - switching to alternative algorithm
            for (unsigned ii = 0; ii < BlockSize; ii += 4) {
                for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
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

template<typename NumType> void
CCTarHor_MultMachine<NumType>::NNBlocks(const unsigned int VectorStartingBlock, const unsigned int VectorBlocksBorder) {
    for (unsigned i = VectorStartingBlock; i < VectorBlocksBorder; i += BlockSize) {
        for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
            for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
                for (unsigned ii = 0; ii < BlockSize; ++ii) {
                    for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                        NumType acc0 = 0;
                        NumType acc1 = 0;
                        NumType acc2 = 0;
                        NumType acc3 = 0;

                        for (unsigned jj = 0; jj < BlockSize; ++jj) {
                            NumType Val = Src2[(i + ii) * Src2SizeOfLine + j + jj];

                            acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                            acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                            acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                            acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                        }


                        Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
                        Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;

                    }
                }
            }

            // Not guaranteed to be divisible by four - switching to alternative algorithm
            for (unsigned ii = 0; ii < BlockSize; ii += 4) {
                for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
                        NumType Val = Src1[(j + jj) * Src1SizeOfLine + k];

                        acc0 += Val * Src2[(i + ii) * Src2SizeOfLine + j + jj];
                        acc1 += Val * Src2[(i + ii + 1) * Src2SizeOfLine + j + jj];
                        acc2 += Val * Src2[(i + ii + 2) * Src2SizeOfLine + j + jj];
                        acc3 += Val * Src2[(i + ii + 3) * Src2SizeOfLine + j + jj];
                    }


                    Target[(i + ii) * TargetSizeOfLine + k] += acc0;
                    Target[(i + ii + 1) * TargetSizeOfLine + k] += acc1;
                    Target[(i + ii + 2) * TargetSizeOfLine + k] += acc2;
                    Target[(i + ii + 3) * TargetSizeOfLine + k] += acc3;

                }
            }
        }

        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize)
            // Doing last run on the not Blocked Base Vectors
        {
            for (unsigned ii = 0; ii < BlockSize; ++ii) {

                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned j = BlocksPerVectorRange; j < Src1Cols; ++j) {
                        NumType Val = Src2[(i + ii) * Src2SizeOfLine + j];

                        acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
                        acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
                        acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
                        acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
                    }


                    Target[(i + ii) * TargetSizeOfLine + k + kk] += acc0;
                    Target[(i + ii) * TargetSizeOfLine + k + kk + 1] += acc1;
                    Target[(i + ii) * TargetSizeOfLine + k + kk + 2] += acc2;
                    Target[(i + ii) * TargetSizeOfLine + k + kk + 3] += acc3;
                }
            }
        }


        for (unsigned ii = 0; ii < BlockSize; ++ii) {
            for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
                NumType acc0 = 0;
                for (unsigned j = BlocksPerVectorRange; j < Src2Rows; ++j) {
                    acc0 += Src1[j * Src1SizeOfLine + k] * Src2[(i + ii) * Src2SizeOfLine + j];
                }
                Target[(i + ii) * TargetSizeOfLine + k] += acc0;
            }
        }
    }
}

template<typename NumType>
void CCTarHor_MultMachine<NumType>::EEFrame() {
    for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
            for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
                        NumType Val = Src2[i * Src2SizeOfLine + j + jj];

                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                    }


                    Target[i * TargetSizeOfLine + k + kk] += acc0;
                    Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                    Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                    Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

                }
            }
        }
    }
}

template<typename NumType>
void CCTarHor_MultMachine<NumType>::ENFrame() {
    for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
            for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
                        NumType Val = Src2[i * Src2SizeOfLine + j + jj];

                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                    }

                    Target[i * TargetSizeOfLine + k + kk] += acc0;
                    Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                    Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                    Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

                }
            }
        }
    }

    for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
        for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
            for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                NumType acc0 = 0;
                NumType acc1 = 0;
                NumType acc2 = 0;
                NumType acc3 = 0;

                for (unsigned j = BlocksPerVectorRange; j < Src2Rows; ++j) {
                    NumType Val = Src2[i * Src2SizeOfLine + j];

                    acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
                    acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
                    acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
                    acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
                }

                Target[i * TargetSizeOfLine + k + kk] += acc0;
                Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                Target[i * TargetSizeOfLine + k + kk + 3] += acc3;

            }
        }
    }

}

template<typename NumType>
void CCTarHor_MultMachine<NumType>::NEFrame() {
    for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
            for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
                        NumType Val = Src2[i * Src2SizeOfLine + (j + jj)];

                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                    }

                    Target[i * TargetSizeOfLine + k + kk] += acc0;
                    Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                    Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                    Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
                }
            }
        }

        for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
            for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
                NumType acc = 0;

                for (unsigned jj = 0; jj < BlockSize; ++jj) {
                    acc += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j + jj];
                }

                Target[i * TargetSizeOfLine + k] += acc;
            }
        }
    }
}

template<typename NumType>
void CCTarHor_MultMachine<NumType>::NNFrame() {
    for (unsigned j = 0; j < BlocksPerVectorRange; j += BlockSize) {
        for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
            for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
                for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                    NumType acc0 = 0;
                    NumType acc1 = 0;
                    NumType acc2 = 0;
                    NumType acc3 = 0;

                    for (unsigned jj = 0; jj < BlockSize; ++jj) {
                        NumType Val = Src2[i * Src2SizeOfLine + j + jj];

                        acc0 += Src1[(j + jj) * Src1SizeOfLine + k + kk] * Val;
                        acc1 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 1] * Val;
                        acc2 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 2] * Val;
                        acc3 += Src1[(j + jj) * Src1SizeOfLine + k + kk + 3] * Val;
                    }

                    Target[i * TargetSizeOfLine + k + kk] += acc0;
                    Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                    Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                    Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
                }
            }
        }

        for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
            for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
                NumType acc = 0;

                for (unsigned jj = 0; jj < BlockSize; ++jj) {
                    acc += Src1[(j + jj) * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j + jj];
                }
                Target[i * TargetSizeOfLine + k] += acc;
            }
        }
    }

    for (unsigned k = 0; k < BlocksPerBaseVectorRange; k += BlockSize) {
        for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
            for (unsigned kk = 0; kk < BlockSize; kk += 4) {
                NumType acc0 = 0;
                NumType acc1 = 0;
                NumType acc2 = 0;
                NumType acc3 = 0;

                for (unsigned j = BlocksPerVectorRange; j < Src2Rows; ++j) {
                    NumType Val = Src2[i * Src2SizeOfLine + j];

                    acc0 += Src1[j * Src1SizeOfLine + k + kk] * Val;
                    acc1 += Src1[j * Src1SizeOfLine + k + kk + 1] * Val;
                    acc2 += Src1[j * Src1SizeOfLine + k + kk + 2] * Val;
                    acc3 += Src1[j * Src1SizeOfLine + k + kk + 3] * Val;
                }

                Target[i * TargetSizeOfLine + k + kk] += acc0;
                Target[i * TargetSizeOfLine + k + kk + 1] += acc1;
                Target[i * TargetSizeOfLine + k + kk + 2] += acc2;
                Target[i * TargetSizeOfLine + k + kk + 3] += acc3;
            }
        }
    }

    for (unsigned i = VectorBlocksRange; i < Src2Cols; ++i) {
        for (unsigned k = BlocksPerBaseVectorRange; k < Src1Rows; ++k) {
            NumType acc = 0;

            for (unsigned j = BlocksPerVectorRange; j < Src2Rows; ++j) {
                acc += Src1[j * Src1SizeOfLine + k] * Src2[i * Src2SizeOfLine + j];
            }
            Target[i * TargetSizeOfLine + k] += acc;
        }
    }
}

// ----------------------------------
// Matrix Transposition Implementation
// ----------------------------------

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

// ---------------------------------
// Matrix Dot product Implementation
// ----------------------------------

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

// ---------------------------------
// Outer Product
// ----------------------------------

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