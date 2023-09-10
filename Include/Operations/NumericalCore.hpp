
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
#include <vector>
#include <queue>

#include "../Management/ThreadSolutions.hpp"

// ------------------------------------------
// Vector & scalar operations
// ------------------------------------------

// Todo replace with own thread structure or else generaly reconsider
template<typename NumType, NumType (*BinOp)(NumType, NumType)>
void ApplyScalarOpOnArray(NumType* const Result, const NumType* const Arg1, const NumType Scalar, const size_t ArraySize){
    #pragma omp parallel for
    for (size_t i = 0; i < ArraySize; ++i){
        Result[i] = BinOp(Arg1[i], Scalar);
    }
}

template<typename NumType, NumType (*BinOp)(NumType, NumType)>
void ApplyArrayOnArrayOp(NumType* const Result, const NumType* const Arg1, const NumType* const Arg2, const size_t ArraySize){
    #pragma omp parallel for
    for (size_t i = 0; i < ArraySize; ++i){
        Result[i] = BinOp(Arg1[i], Arg2);
    }
}

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

// TODO: Explain names

template<typename NumType>
class OuterProductMachine
{
    const NumType* CoefPtr;
    const NumType* VectPtr;
    NumType* const MatC;
    size_t CoefSize;
    size_t VectSize;
    const size_t MatCSoL;

    inline void ProcessCoefBlock(size_t BlockBegin, size_t BlockEnd, size_t VectRange);
    inline void CleanEdges(size_t CleanBegin, size_t CleanOutElementsBegin);

    std::mutex EdgeGuard;
    bool EdgesDone { false };
    size_t ElementsPerThread{};
    size_t VectRangeForThread{};
    size_t CleanBeginForThread{};
    void ThreadInstance(size_t ThreadID);

public:
    OuterProductMachine(const NumType* VectA, const NumType* VectB, NumType* MatC, size_t ASize,
                        size_t BSize, size_t MatCSoL, bool IsHor = false);

    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
    inline void Perform();
};

#ifdef __AVX__

template<>
void OuterProductMachine<double>::ProcessCoefBlock(size_t BlockBegin, size_t BlockEnd, size_t VectRange);

template<>
void OuterProductMachine<double>::CleanEdges(size_t CleanBegin, size_t CleanOutElementsBegin);

#endif

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

    // Threading components
    using P2D = std::tuple<size_t, size_t>;

    std::atomic<bool> WorkDone { false } ;
    std::mutex QueGuard;
    std::queue<P2D> CordQue;

    // TODO: popraw nazwy
    static constexpr size_t RMVKernelHeight() { return 8; }
    static constexpr size_t RMVKernelWidth() { return AVXInfo::GetAVXLength<NumType>(); }
    inline void RMVKernel(size_t HorizontalCord, size_t VerticalCord);


    static constexpr size_t CMVKernelHeight() { return 12 * AVXInfo::GetAVXLength<NumType>(); }
    static constexpr size_t CMVKernelWidth() { return 1; }
    inline void CMVKernel(size_t HorizontalCord, size_t VerticalCord);
    inline void CMVKernelCleaning(size_t HorizontalCord, size_t VerticalCord);
    inline void CMVClean(size_t CleanBegin);

    // Threading parts
    template<void (VMM<NumType>::*Kernel)(size_t, size_t)>
    void ThreadInstance();

    template<void (VMM<NumType>::*Kernel)(size_t, size_t), size_t KernelHeight>
    void ProcessVM(size_t HorizontalRange, size_t VerticalRange);

    template<size_t ThreadCap, size_t (*Decider)(size_t)>
    void ProcessCMV();

    template<size_t ThreadCap, size_t (*Decider)(size_t)>
    void ProcessRMV();

    // Variables used to communicate between threads
    static constexpr size_t GetVectChunkSize(){
        return (CacheInfo::L1Size / 2) / sizeof(NumType);
    }

    static constexpr size_t GetVertMatChunkSize(){
        return (CacheInfo::L2Size * 9) / sizeof(NumType) / 10;
    }

public:
    VMM(const NumType* MatA, const NumType* VectB, NumType* VectC, size_t MatARows, size_t MatACols, size_t MatASoL, bool IsHor);

    template<size_t ThreadCap, size_t (*Decider)(size_t)>
    void PerformVM(){

    }

    template<size_t ThreadCap, size_t (*Decider)(size_t)>
    void PerformMV(){
        if (IsMatHor) ProcessRMV<ThreadCap, Decider>();
        else ProcessCMV<ThreadCap, Decider>();
    }
};

#if defined(__AVX__) && defined(__FMA__)

template<>
constexpr size_t VMM<double>::GetVectChunkSize()
    // 3584 - part of vector hold in L1 cache, should be divisible by 4
{
    return 3584;
}

template<>
constexpr size_t VMM<double>::GetVertMatChunkSize()
    // 240512 - part of the resulting vector used only to save single kernel load - hold entirely in L2 cache
{
    return 240512;
}

template<>
void VMM<double>::RMVKernel(size_t HorizontalCord, size_t VerticalCord);

template<>
void VMM<double>::CMVKernel(size_t HorizontalCord, size_t VerticalCord);

template<>
void VMM<double>::CMVKernelCleaning(size_t HorizontalCord, size_t VerticalCord);

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

// ------------------------------------------
// Outer Product Implementation
// ------------------------------------------

template<typename NumType>
OuterProductMachine<NumType>::OuterProductMachine(const NumType *VectA, const NumType *VectB, NumType *MatC, size_t ASize, size_t BSize,
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
void OuterProductMachine<NumType>::CleanEdges(size_t CleanBegin, size_t CleanOutElements)
    // CleanOutElements - used in case of double cache line alignment. Used in AVX versions
{
    for(size_t  j = 0; j < VectSize; ++j) {
        for (size_t i = CleanBegin; i < CoefSize; ++i) {
            MatC[i * MatCSoL + j] = VectPtr[j] * CoefPtr[i];
        }
    }
}

template<typename NumType>
void OuterProductMachine<NumType>::ProcessCoefBlock(size_t BlockBegin, size_t BlockEnd, size_t VectRange)
    // Coef range (BlockEnd - BlockBegin) must be divisible by ElementsInCacheLine
    // VectRange - Range blockable by size of cache line
{
    for (size_t i = BlockBegin; i < BlockEnd; i += GetCacheLineElem<NumType>()) {
        NumType CoefHolders[GetCacheLineElem<NumType>()];
        // Should be unrolled
        for (size_t k = 0; k < GetCacheLineElem<NumType>(); ++k){
            CoefHolders[k] = CoefPtr[i + k];
        }

        for (size_t j = 0; j < VectSize; ++j) {
            // Should be unrolled
            for (size_t k = 0; k < GetCacheLineElem<NumType>(); ++k){
                MatC[(i + k) * MatCSoL + j] = VectPtr[j] * CoefHolders[k];
            }
        }
    }
}

template<typename NumType>
void OuterProductMachine<NumType>::ThreadInstance(size_t ThreadID) {
    size_t BlockBegin { ThreadID * ElementsPerThread };
    size_t BlockEnd { (ThreadID + 1) * ElementsPerThread };

    ProcessCoefBlock(BlockBegin, BlockEnd,VectRangeForThread);

    if (!EdgesDone && EdgeGuard.try_lock()){
        const size_t BlockableRange { (CoefSize / GetCacheLineElem<NumType>() ) * GetCacheLineElem<double>() };
        if (BlockableRange != CleanBeginForThread) ProcessCoefBlock(CleanBeginForThread, BlockableRange, VectRangeForThread);

        CleanEdges(BlockableRange, VectRangeForThread);
        EdgesDone = true;
        EdgeGuard.unlock();
    }
}

template<typename NumType>
template<size_t ThreadCap, size_t (*Decider)(size_t)>
void OuterProductMachine<NumType>::Perform() {
    const size_t OperationCount { VectSize * CoefSize };

    if (OperationCount < ThreadInfo::ThreadedStartingThreshold) {
        const size_t CoefRange { (CoefSize / GetCacheLineElem<NumType>()) * GetCacheLineElem<NumType>() };
        const size_t VectRange { (VectSize / GetCacheLineElem<NumType>()) * GetCacheLineElem<NumType>() };
        ProcessCoefBlock(0, CoefRange, VectRange);

        if (CoefRange == CoefSize) return;
        CleanEdges(CoefRange, VectRange);
    }
    else{
        const size_t ThreadCount { Decider(OperationCount) };
        ElementsPerThread = ((CoefSize / GetCacheLineElem<NumType>()) / ThreadCount ) * GetCacheLineElem<NumType>();
        VectRangeForThread = (VectSize / GetCacheLineElem<NumType>()) * GetCacheLineElem<NumType>();
        CleanBeginForThread = ThreadCount * ElementsPerThread;

        ExecuteThreads(ThreadCount, &OuterProductMachine<NumType>::ThreadInstance, this);
    }
}

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
void VMM<NumType>::RMVKernel(size_t HorizontalCord, size_t VerticalCord){

}

template<typename NumType>
void VMM<NumType>::CMVKernel(size_t HorizontalCord, size_t VerticalCord){
    static constexpr size_t AccumulatorCount{ 12 };
    size_t Range { std::min(MatACols, HorizontalCord + GetVectChunkSize()) };

    auto ProcessSingleAccPack = [&](size_t Offset) -> void {
        NumType AccumulatorRegisters[AccumulatorCount] { NumType() };

        for (size_t i = HorizontalCord; i < Range; i += CMVKernelWidth()) {
            NumType CoefBuff = VectB[i];

            // Should be unrolled
            for (size_t j = 0; j < AccumulatorCount; ++j) {
                AccumulatorRegisters[j] += CoefBuff * MatA[i * MatASoL + VerticalCord + j + Offset];
            }
        }

        // Should be unrolled
        for (size_t i = 0; i < AccumulatorCount; ++i) {
            VectC[VerticalCord + i + Offset] += AccumulatorRegisters[i];
        }
    };

    // Should be unrolled
    for(size_t i = 0; i < AVXInfo::GetAVXLength<NumType>(); ++i){
        ProcessSingleAccPack(12 * i);
    }
}

template<typename NumType>
void VMM<NumType>::CMVKernelCleaning(size_t HorizontalCord, size_t VerticalCord) {
    static constexpr size_t MaxRegisterCount { 8 }; // because x64 has only 16 registers, 8 is a universal number,
    // allowing to divide different sizes of NumType without any cleaning needed
    size_t Range { std::min(MatACols, HorizontalCord + GetVectChunkSize()) };
    static constexpr size_t AccumulatorCount { []() constexpr -> size_t {
        if constexpr ( GetCacheLineElem<NumType>() > MaxRegisterCount){
            return MaxRegisterCount;
        }
        else return GetCacheLineElem<NumType>();
    }() };

    auto ProcessSingleAccPack = [&](size_t Offset) -> void {
        NumType AccumulatorRegisters[AccumulatorCount] { NumType() };

        for (size_t i = HorizontalCord; i < Range; i += CMVKernelWidth()) {
            NumType CoefBuff = VectB[i];

            // Should be unrolled
            for (size_t j = 0; j < AccumulatorCount; ++j) {
                AccumulatorRegisters[j] += CoefBuff * MatA[i * MatASoL + VerticalCord + j + Offset];
            }
        }

        // Should be unrolled
        for (size_t i = 0; i < AccumulatorCount; ++i) {
            VectC[VerticalCord + i + Offset] += AccumulatorRegisters[i];
        }
    };

    if constexpr(GetCacheLineElem<NumType>() > MaxRegisterCount){
        const size_t AccPackRange {GetCacheLineElem<NumType>() / MaxRegisterCount }; // divides CacheLine to chunks fitting in registers

        // Should be unrolled
        for (size_t i = 0 ; i < AccPackRange; ++i){
            ProcessSingleAccPack(AccumulatorCount * i);
        }
    }
    else{
        ProcessSingleAccPack(0);
    }

}

template<typename NumType>
void VMM<NumType>::CMVClean(size_t CleanBegin) {
    for (size_t j = 0; j < MatACols; j += GetVectChunkSize()) {
        for (size_t i = CleanBegin; i < MatARows; i += GetCacheLineElem<NumType>()) {
            CMVKernelCleaning(j, i);
        }
    }
}

template<typename NumType>
template<void (VMM<NumType>::*Kernel)(size_t, size_t)>
void VMM<NumType>::ThreadInstance() {
    while(!WorkDone || !CordQue.empty()){
        QueGuard.lock();

        if (CordQue.empty()) {
            QueGuard.unlock();
            continue;
        }

        P2D TargetCord = CordQue.front();
        CordQue.pop();
        QueGuard.unlock();

        (this->*Kernel)(std::get<0>(TargetCord), std::get<1>(TargetCord));
    }
}

template<typename NumType>
template<void (VMM<NumType>::*Kernel)(size_t, size_t), size_t KernelHeight>
void VMM<NumType>::ProcessVM(size_t HorizontalRange, size_t VerticalRange) {
    for (size_t i = 0; i < HorizontalRange; i += GetVectChunkSize()){
        for(size_t j = 0; j < VerticalRange; j += GetVertMatChunkSize()){
            const size_t Range = std::min(VerticalRange, j + GetVertMatChunkSize());
            #pragma omp parallel for
            for(size_t jj = j; jj < Range; jj += KernelHeight){
                (this->*Kernel)(i, jj);
            }
        }
    }
}

// TODO: THERE IS ULTRA RARE THREAD RACE DO IT WITH COND VARIABLES INSTEAD OF SLEEP
template<typename NumType>
template<size_t ThreadCap, size_t (*Decider)(size_t)>
void VMM<NumType>::ProcessCMV() {
    const size_t VerRange { (MatARows / CMVKernelHeight()) * CMVKernelHeight() };
    const size_t OperationCount { MatARows * MatACols };

    if (OperationCount < ThreadInfo::ThreadedStartingThreshold){
        ProcessVM<&VMM<NumType>::CMVKernel, CMVKernelHeight()>(MatACols, VerRange);
        CMVClean(VerRange);
    }
    else{
        const size_t ThreadAmount { Decider(OperationCount) };
        ThreadPackage& Threads = ExecuteThreadsWNJoining(ThreadAmount,
                                                         &VMM<NumType>::ThreadInstance<VMM<NumType>::CMVKernel>, this);

        for (size_t i = 0; i < MatACols; i += GetVectChunkSize()){
            while(!CordQue.empty()){
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            for(size_t j = 0; j < VerRange; j += GetVertMatChunkSize()){
                const size_t Range = std::min(VerRange, j + GetVertMatChunkSize());
                for(size_t jj = j; jj < Range; jj += CMVKernelHeight()){
                    QueGuard.lock();
                    CordQue.emplace(i, jj);
                    QueGuard.unlock();
                }
            }
        }
        WorkDone = true;
        CMVClean(VerRange);
        JoinThreads(ThreadAmount, Threads);
    }
}

template<typename NumType>
template<size_t ThreadCap, size_t (*Decider)(size_t)>
void VMM<NumType>::ProcessRMV() {
    ProcessVM<&VMM<NumType>::RMVKernel, RMVKernelHeight()>(MatACols, MatARows);
}

#endif // PARALLELNUM_NUMERICAL_CORE_H_