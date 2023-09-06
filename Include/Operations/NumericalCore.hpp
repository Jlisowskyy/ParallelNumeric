
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

#include "../Management/ThreadSolutions.hpp"
//#include "MatrixMultiplicationSolutions.hpp"

// ------------------------------------------
// Sum of matrices functions
// ------------------------------------------

template<typename NumType>
class MatrixSumMachine{
    const NumType* const MatA;
    const NumType* const MatB;
    NumType* const MatC;

    const size_t Rows;
    const size_t Cols;
    const size_t Size; // Used for aligned arrays
    size_t DimToDivide{};
    size_t ElementsPerThread{};
    size_t CleanBegin{};

    const size_t MatASoL;
    const size_t MatBSoL;
    const size_t MatCSoL;

    void (MatrixSumMachine<NumType>::*BlockFunc)(size_t, size_t) {};
    void (MatrixSumMachine<NumType>::*FrameFunc)(size_t) {};
    std::mutex FrameGuard;
    bool FrameDone = false;

    void AlignedArrays(size_t Begin, size_t End);
    void AlignedArraysCleaning(size_t Begin);

    void RCBlockedByCols(size_t StartCol, size_t StopCol);
    void RCBlockedByRows(size_t StartRow, size_t StopRow);
    void CRBlockedByCols(size_t StartCol, size_t StopCol);
    void CRBlockedByRows(size_t StartRow, size_t StopRow);

    void RCBlockedByColsFrame(size_t StartCol);
    void RCBlockedByRowsFrame(size_t StartRow);
    void CRBlockedByColsFrame(size_t StartCol);
    void CRBlockedByRowsFrame(size_t StartRow);

    void ThreadInstance(size_t ThreadID);

    inline void UnalignedArraysMatchingFunctions(bool IsAHor, bool IsBHor, bool IsCHor);
public:
    MatrixSumMachine(const NumType* MatA, const NumType* MatB, NumType* MatC, size_t Rows, size_t Cols, size_t Size,
                     size_t MatASoL, size_t MatBSoL, size_t MatCSoL, bool IsAHor, bool IsBHor, bool IsCHor);

    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
    inline void Perform();
};

#ifdef __AVX__

template<>
void MatrixSumMachine<double>::AlignedArrays(size_t Begin, size_t End);

template<>
void MatrixSumMachine<float>::AlignedArrays(size_t Begin, size_t End);

#endif // __AVX__

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

// TODO: Przetestuj template z roznymi ilosciami 'rejestrow'

template<typename NumType>
class DotProductMachine{
    static constexpr size_t GetKernelSize()
        // Neutral choice used only in unsupported types
    {
        return 8;
    }

#if defined(__AVX__) && defined(__FMA__)
    static constexpr size_t AVXAccumulators = 12;
        //
#endif

    const NumType* const APtr;
    const NumType* const BPtr;
    const size_t Size;

    size_t ElementsPerThread{};
    void ThreadInstance(size_t ThreadID, NumType* ReturnVal);
    NumType DotProductAligned(size_t Begin, size_t End);
    inline NumType DotProductCleaning(size_t BeginOfCleaning);
    inline NumType DotProduct();
public:
    DotProductMachine(const NumType* const APtr, const NumType* const BPtr, const size_t Size):
        APtr{ APtr }, BPtr{ BPtr }, Size{ Size } {}

    template<size_t ThreadCap = 8, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
    inline NumType Perform();
};

template<typename NumType>
template<size_t ThreadCap, size_t (*Decider)(size_t)>
NumType DotProductMachine<NumType>::Perform() {
    if (Size < ThreadInfo::ThreadedStartingThreshold) {
        return DotProduct();
    }
    else {
        NumType ThreadResult[ThreadCap]{ NumType{} };
        const size_t ThreadAmount = Decider(Size * 2);
        ElementsPerThread = (Size / (ThreadAmount * GetKernelSize())) * GetKernelSize();

        ExecuteThreadsWOutput(ThreadResult, ThreadAmount, &DotProductMachine<NumType>::ThreadInstance, this);

        NumType RetVal{};
        for(auto& Iter : ThreadResult) RetVal += Iter;
        return RetVal + DotProductCleaning(ElementsPerThread * ThreadAmount);
    }
}

#ifdef __AVX__

template<>
constexpr size_t DotProductMachine<double>::GetKernelSize() {
    return AVXAccumulators * AVXInfo::f64Cap;
}

template<>
double DotProductMachine<double>::DotProductAligned(size_t Begin, size_t End);

#endif // __AVX__

// ------------------------------------------
// Outer Product
// ------------------------------------------

template<typename NumType>
class OPM
    // Outer Product Machine
    // TODO Optimize for L3 - long vectors
{
    static constexpr size_t ElementsPerCacheLine = CacheInfo::LineSize / sizeof(NumType);

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


template<typename NumType>
void MatrixSumMachine<NumType>::UnalignedArraysMatchingFunctions(bool IsAHor, bool IsBHor, bool IsCHor) {
    static constexpr size_t InequalityThreshold = 50;
    using MachT = MatrixSumMachine<NumType>;

    if (IsAHor){
        if (Rows > InequalityThreshold * Cols){
            BlockFunc = &MachT::RCBlockedByRows;
            FrameFunc = &MachT::RCBlockedByRowsFrame;
            DimToDivide = Rows;
        }
        else{
            BlockFunc = &MachT::RCBlockedByCols;
            FrameFunc = &MachT::RCBlockedByColsFrame;
            DimToDivide = Cols;
        }
    }
    else{
        if (Rows > InequalityThreshold * Cols){
            BlockFunc = &MachT::CRBlockedByRows;
            FrameFunc = &MachT::CRBlockedByRowsFrame;
            DimToDivide = Rows;
        }
        else{
            BlockFunc = &MachT::CRBlockedByCols;
            FrameFunc = &MachT::CRBlockedByColsFrame;
            DimToDivide = Cols;
        }
    }
}

template<typename NumType>
MatrixSumMachine<NumType>::MatrixSumMachine(const NumType * const MatA, const NumType * const MatB, NumType * const MatC,
                                            const size_t Rows, const size_t Cols, const size_t Size, const size_t MatASoL,
                                            const size_t MatBSoL, size_t MatCSoL, const bool IsAHor, const bool IsBHor, const bool IsCHor):
        MatA{ MatA }, MatB{ MatB }, MatC{ MatC }, Rows{ Rows }, Cols{ Cols },  MatASoL { MatASoL }, MatBSoL{ MatBSoL },
        MatCSoL{ MatCSoL }, Size { Size }
{
    if (IsAHor != IsBHor){
        UnalignedArraysMatchingFunctions(IsAHor, IsBHor, IsCHor);
    }
    else{
        BlockFunc = &MatrixSumMachine<NumType>::AlignedArrays;
        FrameFunc = &MatrixSumMachine<NumType>::AlignedArraysCleaning;
        DimToDivide = Size;
    }
}

template<typename NumType>
template<size_t ThreadCap, size_t (*Decider)(size_t)>
void MatrixSumMachine<NumType>::Perform() {
    if (Size < ThreadInfo::ThreadedStartingThreshold){
        const size_t Range = (DimToDivide / GetCacheLineElem<NumType>()) * GetCacheLineElem<NumType>();
        (this->*BlockFunc)(0, Range);
        (this->*FrameFunc)(Range);
    }
    else{
        const size_t ThreadAmount = Decider(Size);
        ElementsPerThread = ((DimToDivide / GetCacheLineElem<NumType>()) / ThreadAmount) * GetCacheLineElem<NumType>();
        CleanBegin = ElementsPerThread * ThreadAmount;

        ExecuteThreads(ThreadAmount, &MatrixSumMachine<NumType>::ThreadInstance, this);
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::ThreadInstance(const size_t ThreadID) {
    const size_t StartPos = ThreadID * ElementsPerThread;
    const size_t StopPos = (ThreadID + 1) * ElementsPerThread;

    (this->*BlockFunc)(StartPos, StopPos);

    if (!FrameDone && FrameGuard.try_lock()){
        (this->*FrameFunc)(CleanBegin);
        FrameGuard.unlock();
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::AlignedArrays(size_t Begin, size_t End){
    for (size_t i = Begin; i < End; ++i)
        MatC[i] = MatA[i] + MatB[i];
}

template<typename NumType>
void MatrixSumMachine<NumType>::AlignedArraysCleaning(size_t Begin) {
    for (size_t i = Begin; i < Size; ++i)
        MatC[i] = MatA[i] + MatB[i];
}

template<typename NumType>
void MatrixSumMachine<NumType>::RCBlockedByCols(size_t StartCol, size_t StopCol) {
    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t RowsRange = ((size_t)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (size_t j = StartCol; j < StopCol; j += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    MatC[(i + z) * MatCSoL + j + k] = MatA[(i + z) * MatASoL + j + k] + MatB[(j + k) * MatBSoL + i + z];
                }
            }
        }
    }

    for (size_t i = StartCol; i < StopCol; ++i) {
        for (size_t j = RowsRange; j < Rows; ++j) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                MatC[j * MatCSoL + i + k] = MatA[j * MatASoL + i + k] + MatB[(i + k) * MatBSoL + j];
            }
        }
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::RCBlockedByRows(size_t StartRow, size_t StopRow) {
    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t j = StartRow; j < StopRow; j += ElementsInCacheLine) {
        for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    MatC[(j + k) * MatCSoL + i + z] = MatA[(j + k) * MatASoL + i + z] + MatB[(i + z) * MatBSoL + j + k];
                }
            }
        }
    }

    for (unsigned j = StartRow; j < StopRow; ++j) {
        for (unsigned i = ColsRange; i < Cols; ++i) {
            MatC[j * MatCSoL + i] = MatA[j * MatASoL + i] + MatB[i * MatBSoL + j];
        }
    }

}

template<typename NumType>
void MatrixSumMachine<NumType>::CRBlockedByCols(size_t StartCol, size_t StopCol) {
    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t RowsRange = ((size_t) (Rows / ElementsInCacheLine)) * ElementsInCacheLine;
    for (size_t j = StartCol; j < StopCol; j += ElementsInCacheLine) {
        for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    MatC[(j + k) * MatCSoL + i + z] = MatA[(j + k) * MatASoL + i + z] + MatB[(i + z) * MatBSoL + j + k];
                }
            }
        }
    }

    for (size_t j = StartCol; j < StopCol; ++j) {
        for (size_t z = RowsRange; z < Rows; ++z) {
            MatC[j * MatCSoL + z] = MatA[j * MatASoL + z] + MatB[z * MatBSoL + j];
        }
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::CRBlockedByRows(size_t StartRow, size_t StopRow) {
    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = StartRow; j < StopRow; j += ElementsInCacheLine) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k) {
                for (size_t z = 0; z < ElementsInCacheLine; ++z) {
                    MatC[(i + k) * MatCSoL + j + z] = MatA[(i + k) * MatASoL + j + z] + MatB[(j + z) * MatBSoL + i + k];
                }
            }
        }
    }

    for (size_t i = StartRow; i < StopRow; i += ElementsInCacheLine) {
        for (size_t z = ColsRange; z < Cols; ++z) {
            for (size_t k = 0; k < ElementsInCacheLine; ++k)
                MatC[z * MatCSoL + i + k] = MatA[z * MatASoL + i + k] + MatB[(i + k) * MatBSoL + z];
        }
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::RCBlockedByColsFrame(size_t StartCol) {
    if (StartCol == Cols) return;

    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t RowsRange = ((size_t)(Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i+= ElementsInCacheLine) {
        for (size_t k = StartCol; k < Cols; ++k) {
            for (size_t j = 0; j < ElementsInCacheLine; ++j) {
                MatC[(i + j) * MatCSoL + k] = MatA[(i + j) * MatASoL + k] + MatB[k * MatBSoL + i + j];
            }
        }
    }

    for (size_t i = RowsRange; i < Rows; ++i) {
        for (size_t j = StartCol; j < Cols; ++j)
            MatC[i * MatCSoL + j] = MatA[i * MatCSoL + j] + MatB[j * MatCSoL + i];
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::RCBlockedByRowsFrame(size_t StartRow) {
    if (StartRow == Rows) return;

    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = 0; j < ElementsInCacheLine; ++j) {
            for (size_t k = StartRow; k < Rows; ++k) {
                MatC[(i + j) * MatCSoL + k] = MatA[(i + j) * MatASoL + k] + MatB[k * MatBSoL + i + j];
            }
        }
    }

    for (size_t i = ColsRange; i < Cols; ++i) {
        for (size_t k = StartRow; k < Rows; ++k) {
            MatC[i * MatCSoL + k] = MatA[i * MatASoL + k] + MatB[k * MatBSoL + i];
        }
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::CRBlockedByColsFrame(size_t StartCol) {
    if (StartCol == Cols) return;

    static constexpr size_t ElementsInCacheLine = GetCacheLineElem<NumType>();
    const size_t RowsRange = ((size_t) (Rows / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < RowsRange; i += ElementsInCacheLine) {
        for (size_t k = StartCol; k < Cols; ++k) {
            for (size_t j = 0; j < ElementsInCacheLine; ++j) {
                MatC[k * MatCSoL + i + j] = MatA[k * MatASoL + i + j] + MatB[(i + j) * MatBSoL + k];
            }
        }
    }

    for (size_t i = RowsRange; i < Rows; ++i) {
        for (size_t k = StartCol; k < Cols; ++k) {
            MatC[k * MatCSoL + i] = MatA[k * MatASoL + i] + MatB[i * MatBSoL + k];
        }
    }
}

template<typename NumType>
void MatrixSumMachine<NumType>::CRBlockedByRowsFrame(size_t StartRow) {
    if (StartRow == Rows) return;

    const size_t ElementsInCacheLine = CacheInfo::LineSize / sizeof(NumType);
    const size_t ColsRange = ((size_t)(Cols / ElementsInCacheLine)) * ElementsInCacheLine;

    for (size_t i = 0; i < ColsRange; i += ElementsInCacheLine) {
        for (size_t j = 0; j < ElementsInCacheLine; ++j) {
            for (size_t k = StartRow; k < Rows; ++k) {
                MatC[(i + j) * MatCSoL + k] = MatA[(i + j) * MatASoL + k] + MatB[k * MatBSoL + i + j];
            }
        }
    }

    for (size_t i = ColsRange; i < Cols; ++i) {
        for (size_t k = StartRow; k < Rows; ++k) {
            MatC[i * MatCSoL + k] = MatA[i * MatASoL + k] + MatB[k * MatBSoL + i];
        }
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
// Vector Dot Product Implementation
// ------------------------------------------

template<typename NumType>
NumType DotProductMachine<NumType>::DotProductAligned(size_t Begin, size_t End) {
    NumType Accumulators[GetKernelSize()] {NumType{} };

    for(size_t i = Begin; i < End; i += GetKernelSize()){
        for(size_t j = 0; j < GetKernelSize(); ++j){
            Accumulators[j] += APtr[i + j] * BPtr[i + j];
        }
    }

    NumType AccSum {};
    for (auto& Iter : Accumulators) AccSum += Iter;
    return AccSum;
}

template<typename NumType>
inline NumType DotProductMachine<NumType>::DotProductCleaning(size_t BeginOfCleaning) {
    NumType RetVal{};
    for (size_t i = BeginOfCleaning; i < Size; ++i){
        RetVal += APtr[i] * BPtr[i];
    }
    return RetVal;
}

template<typename NumType>
inline NumType DotProductMachine<NumType>::DotProduct()
    // Works for unsupported types or in case of lack of avx
{
    const size_t Range = (Size / GetKernelSize()) * GetKernelSize();
    return DotProductAligned(0, Range) + DotProductCleaning(Range);
}

template<typename NumType>
void DotProductMachine<NumType>::ThreadInstance(size_t ThreadID, NumType* ReturnVal) {
    size_t Begin = ThreadID * ElementsPerThread;
    size_t End = (ThreadID + 1) * ElementsPerThread;

    *ReturnVal = DotProductAligned(Begin, End);
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
            __m256d VectA1 = _mm256_load_pd(VectPtrIter + AVXInfo::f64Cap);
            VectPtrIter += ElementsPerCacheLine;

            double *TargetFirstPtr0 = MatC + i * MatCSoL + j;
            double *TargetSecondPtr0 = MatC + i * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr1 = MatC + (i + 2) * MatCSoL + j;
            double *TargetSecondPtr1 = MatC + (i + 2) * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr2 = MatC + (i + 4) * MatCSoL + j;
            double *TargetSecondPtr2 = MatC + (i + 4) * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr3 = MatC + (i + 6) * MatCSoL + j;
            double *TargetSecondPtr3 = MatC + (i + 6) * MatCSoL + j + AVXInfo::f64Cap;
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
        __m256d VectSecond = _mm256_load_pd(VectPtr + j + AVXInfo::f64Cap);

        for (size_t i = 0; i < CleaningRange; i++)
#define CleaningTargetUpper MatC + (i + CoefRange) * MatCSoL + j
#define CleaningTargetLower MatC + (i + CoefRange) * MatCSoL + j + AVXInfo::f64Cap
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