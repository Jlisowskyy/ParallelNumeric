
// Author: Jakub Lisowski

#ifndef PARALLELNUM_MATRIX_H_
#define PARALLELNUM_MATRIX_H_

#include <cmath>
#include <thread>
#include <iostream>
#include <functional>
#include <utility>
#include <iomanip>
#include <immintrin.h>
#include <exception>

#include "Vector.hpp"
#include "../Operations/MatrixMultiplication.hpp"

template<typename NumType>
class Matrix1 : public Vector<NumType>
{
	//FixedSize - not expandable
	static const unsigned ElementsPerCacheLine = CACHE_LINE / sizeof(NumType);

	unsigned Rows, Cols; 
	// Contains only information about matrix size

	unsigned ElementsPerLine{}, Lines{};
	// Contains only information about data storing and alignment

	unsigned long MatrixSize; 
	// Store only data about mathematical matrix dimensions not actual on hw size

	unsigned OffsetPerLine{};
	// Contains information how many data cells in memory are shifted to ensure data line alignment

	unsigned SizeOfLine{};
	// Contains information about actual hw size of each line different from ElementsPerLine by alignment
	
	bool IsMemoryPacked = false; 
	// Memory optimizations for edge cases

	using Vector<NumType>::Array;
	using Vector<NumType>::IsHorizontal;

	using Vector<NumType>::MM;
	// Probably used in future to synchronize resource management

	using Vector<NumType>::Size;
	// actual on hw size expressed by number of held NumType variables inside the array

	using Vector<NumType>::CheckForIncorrectSize;
	NumType& (Matrix1::* AccessFunc)(const unsigned&, const unsigned&);
	const NumType& (Matrix1::* AccessFuncConst)(const unsigned&, const unsigned&) const;

	void AbandonIfVector() const {
		if (Rows == 1 || Cols == 1) {
			exit(0xf2);
		}
	}

	void PerformSanityChecks() const {
		AbandonIfVector();

		if (MatrixSize == 0) {
			exit(0xf1);
		}
	}

	void OptimizeResourceManagement(NumType* InitVal = nullptr);
	void SetupAccess();
	void MoveFromPointer(NumType* Src);
public:
	Matrix1(unsigned NNSize, bool ByRow = false, ResourceManager* MM = DefaultMM) noexcept;
	Matrix1(unsigned Rows, unsigned Cols, bool ByRow = false, ResourceManager* MM = DefaultMM) noexcept;
	Matrix1(unsigned NNSize, NumType InitVal, bool ByRow = false, ResourceManager* MM = DefaultMM) noexcept;
	Matrix1(unsigned Rows, unsigned Cols, NumType InitVal, bool ByRow = false, ResourceManager* MM = DefaultMM) noexcept;
	Matrix1(const Matrix1& Target) noexcept;
	Matrix1(Matrix1&& Target) noexcept;
	Matrix1(std::initializer_list<std::initializer_list<NumType>> Init, bool ByRow = false, ResourceManager* MM = DefaultMM) noexcept;
	Matrix1(unsigned NNSize, const NumType* Init, bool ByRow = false, ResourceManager* MM = DefaultMM);
	Matrix1(unsigned Rows, unsigned Cols, const NumType* Init, bool ByRow = false, ResourceManager* MM = DefaultMM);

	// Data accessing operators

	inline NumType& AccessByRow(const unsigned& Row, const unsigned& Col) {
		return Array[Row * SizeOfLine + Col];
	}

	inline NumType& AccessByCol(const unsigned& Row, const unsigned& Col) {
		return Array[Col * SizeOfLine + Row];
	}

	inline const NumType& AccessByRowConst(const unsigned& Row, const unsigned& Col) const {
		return Array[Row * SizeOfLine + Col];
	}

	inline const NumType& AccessByColConst(const unsigned& Row, const unsigned& Col) const {
		return Array[Col * SizeOfLine + Row];
	}

	inline const NumType& operator()(const unsigned& Row, const unsigned& Col) const {
		return (this->*AccessFuncConst)(Row, Col);
	}

	inline NumType& operator()(const unsigned& Row, const unsigned& Col) {
		return (this->*AccessFunc)(Row, Col);
	}

	inline NumType& operator[](const unsigned& Index) {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline const NumType& operator[](const unsigned& Index) const {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline unsigned GetRows() const { return Rows; }
	inline unsigned GetCols() const { return Cols; }
	inline std::pair<unsigned, unsigned> GetDim() { return std::make_pair(Rows, Cols); }

#ifdef DEBUG_
	virtual bool CheckForIntegrity(NumType Val, bool verbose);
	virtual bool CheckForIntegrity(NumType* Val, bool verbose);
#endif // DEBUG_

	Matrix1& operator=(const Matrix1& x);
	Matrix1& operator=(Matrix1&& x) noexcept;

	friend std::ostream& operator<<(std::ostream& out, Matrix1& MyMatrix){
        int MaxMatrixCols = (FindConsoleWidth() - 2) / 6 > 0 ? (FindConsoleWidth() - 2) / 6 : 0;

        out << std::fixed << std::setprecision(3);

        if (MyMatrix.Cols <= (unsigned)MaxMatrixCols) {
            MyMatrix.PrintWhole(out);
        }
        else {
            MyMatrix.PrintPartitioned(out, MaxMatrixCols);
        }

        return out;
    }

private:
	void PrintWhole(std::ostream& out);
	void PrintPartitioned(std::ostream& out, int MaxMatrixCols);

public:
	// OPERATIONS ON MATRICES

	Matrix1 GetTransposed() const;

	const Matrix1& Transpose()
		// Uses naive algorithm
	{
		return *this = GetTransposed(*this);
	}

    template<unsigned ThreadCap,unsigned (*Decider)(unsigned long long)>
    friend Matrix1 MatrixSumSameAccess(const Matrix1 &a, const Matrix1 &b)
        // Handles the case when both matrices are stored in the same way
    {
        Matrix1 RetVal(a.Rows, a.Cols, a.IsHorizontal);

        if (a.Size < ThreadedStartingThreshold) {
            MatrixSumHelperAlignedArrays(RetVal.Array, a.Array, b.Array, a.Size);
            return RetVal;
        }

        unsigned ThreadAmount = Decider(a.Size);
        ThreadPackage Threads = ResourceManager::GetThreads();
        unsigned i;
        unsigned long ElementsPerThread = a.Size / (unsigned long)ThreadAmount;

        for (i = 0; i < ThreadAmount - 1; ++i) {
            unsigned long offset = i * ElementsPerThread;
            Threads.Array[i] = new std::thread(MatrixSumHelperAlignedArrays<NumType>,
                                               RetVal.Array + offset, a.Array + offset, b.Array + offset, ElementsPerThread
            );
        }

        unsigned long offset = i * ElementsPerThread;
        Threads.Array[i] = new std::thread(MatrixSumHelperAlignedArrays<NumType>,
                                           RetVal.Array + offset, a.Array + offset, b.Array + offset, a.Size - offset
        );

        for (i = 0; i < ThreadAmount; ++i) {
            Threads.Array[i]->join();
            delete Threads.Array[i];
        }

        Threads.Release();
        return RetVal;
    }

#define InequalityThreshold 2
// TODO: replace function pointers and create unversal threaded procedure
    template<unsigned ThreadCap, unsigned (*Decider)(unsigned long long)>
    friend Matrix1 MatrixSumDiffAccess(const Matrix1 &a, const Matrix1 &b) {
        Matrix1 RetVal(a.Rows, a.Cols, a.IsHorizontal);
        void (*Func)(NumType*, const NumType*, const NumType*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
        void (*FrameFunc)(NumType*, const NumType*, const NumType*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
        unsigned ElementsPerThread, DimToDivide, CoDim;

        if (!a.IsHorizontal && b.IsHorizontal)
            // Left matrix is col stored and right is row stored
        {
            if (a.Rows > InequalityThreshold * a.Cols)
                // Choosing a preferable situation for block chunking
            {
                Func = &MatrixSumHelperNotAlignedArrays_CR_DivByRows<NumType>;
                FrameFunc = &MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame<NumType>;
                DimToDivide = a.Rows; CoDim = a.Cols;
            }
            else {
                Func = &MatrixSumHelperNotAlignedArrays_CR_DivByCols<NumType>;
                FrameFunc = &MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame<NumType>;
                DimToDivide = a.Cols; CoDim = a.Rows;
            }
        }
        else
            // Opposite situation
        {
            if (a.Rows > InequalityThreshold * a.Cols) {
                Func = &MatrixSumHelperNotAlignedArrays_RC_DivByRows<NumType>;
                FrameFunc = &MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame<NumType>;
                DimToDivide = a.Rows; CoDim = a.Cols;
            }
            else {
                Func = &MatrixSumHelperNotAlignedArrays_RC_DivByCols<NumType>;
                FrameFunc = &MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame<NumType>;
                DimToDivide = a.Cols; CoDim = a.Rows;
            }

        }

        if (a.Size < ThreadedStartingThreshold){
            ElementsPerThread = (DimToDivide / CACHE_LINE) * CACHE_LINE;

            Func(RetVal.Array, a.Array, b.Array, 0, ElementsPerThread, CoDim,
                 RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine);
            FrameFunc(RetVal.Array, a.Array, b.Array, ElementsPerThread, DimToDivide, CoDim,
                      RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine);

            return RetVal;
        }

        unsigned ThreadAmount = Decider(a.Size);
        ThreadPackage& Threads = ResourceManager::GetThreads();
        unsigned i;
        ElementsPerThread = (DimToDivide / ThreadAmount);

        for (i = 0; i < ThreadAmount; ++i) {
            unsigned Start = i * ElementsPerThread;
            unsigned Stop = (i + 1) * ElementsPerThread;

            Threads.Array[i] = new std::thread(Func, RetVal.Array, a.Array, b.Array,
                                               Start, Stop, CoDim, RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine
            );
        }

        for (unsigned j = 0; j < ThreadAmount; ++j) {
            Threads.Array[j]->join();
            delete Threads.Array[j];
        }

        unsigned Start = i * ElementsPerThread;
        FrameFunc(RetVal.Array, a.Array, b.Array, Start, DimToDivide, CoDim,
                  RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine);

        Threads.Release();
        return RetVal;
    }

    template<unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LogarithmicThreads<ThreadCap>>
	friend Matrix1 operator+(const Matrix1& a, const Matrix1& b)
        // Threaded sum of matrices
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
#ifdef _MSC_VER
            throw std::exception("Wrong matrix size");
#else
            throw std::exception();
#endif

        if (a.IsHorizontal != b.IsHorizontal) {
            std::cerr << "Not matching Accessing types matrices - much slower operations\n\n";
        }

        // Solution A
        if (a.IsHorizontal == b.IsHorizontal )
            return MatrixSumSameAccess<ThreadCap, Decider>(a, b);
        else
            return MatrixSumDiffAccess<ThreadCap, Decider>(a, b);
    }
private:
    template<typename NumT>
    friend inline GPMM<NumT> GetMultMachine(const Matrix1<NumT>& A, const Matrix1<NumT>& B, const Matrix1<NumT>& C){
        return GPMM<NumType>(A.Array, B.Array, C.Array, A.Rows, A.Cols, B.Cols, A.SizeOfLine, B.SizeOfLine, C.SizeOfLine);
    }

public:
    template<unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LogarithmicThreads<ThreadCap>>
	friend Matrix1 operator*(const Matrix1& A, const Matrix1& B){
        if (A.Cols != B.Rows)
            throw std::runtime_error("Not able to perform matrix multiplication - wrong matrix sizes\n");

        Matrix1 RetVal(A.Rows, B.Cols, (NumType)0);
        GPMM<NumType> MultMachine = GetMultMachine(A,B,RetVal);
        MultMachine.CCPerform();

        return RetVal;
    }

	Matrix1 GetModified(void (Vector<NumType>::* func)()) {
		Matrix1 RetVal = *this;
		RetVal.*func();
		return RetVal;
	}

	Matrix1 GetModified(NumType(*func)(NumType x)) {
		Matrix1 RetVal = *this;

		NumType* Dst = RetVal.Array;
		for (unsigned long i = 0; i < Size; ++i) {
			Dst[i] = func(Dst[i]);
		}

		return RetVal;
	}
};

template<typename NumType>
void Matrix1<NumType>::OptimizeResourceManagement(NumType *InitVal)
// Find optimal way to store the data, then prepares
// Arrays to be used efficiently, needs variable Rows and Cols to operate
{
    unsigned long PerCacheLine = CACHE_LINE / sizeof(NumType);
    unsigned long ElementsOnLastCacheLine = ElementsPerLine % PerCacheLine;
    OffsetPerLine = ElementsOnLastCacheLine == 0 ? 0 : (unsigned) (PerCacheLine - ElementsOnLastCacheLine);
    SizeOfLine = ElementsPerLine + OffsetPerLine;
    unsigned long ExpectedAlignedSize = Lines * SizeOfLine;
    unsigned long MemoryEnlargementInBytes = OffsetPerLine * Lines * (unsigned long)sizeof(NumType);

    if (MemoryEnlargementInBytes > (GB / 4) && (double)ExpectedAlignedSize > 1.2 * (double)MatrixSize )
        // Stores data partitioned to prevent cache lines overlapping
        // in case when alignment could take more than 0.25Gb
        // Temporary barrier to be reconsidered in the future
    {
        IsMemoryPacked = true;
        OffsetPerLine = 0;
        SizeOfLine = ElementsPerLine;

        *((Vector<NumType>*)this) = InitVal == nullptr ?
                                    Vector<NumType>(MatrixSize, IsHorizontal) :
                                    Vector<NumType>(MatrixSize, *InitVal, IsHorizontal);

        return;
    }

    // Standard data storing with aligned every line of matrix
    *((Vector<NumType>*)this) = InitVal == nullptr ?
                                Vector<NumType>(ExpectedAlignedSize, IsHorizontal) :
                                Vector<NumType>(ExpectedAlignedSize, *InitVal, IsHorizontal);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int Rows, unsigned int Cols, const NumType *Init, bool ByRow, ResourceManager *MM) :
        Vector<NumType>{ ByRow, MM }, Rows{ Rows }, Cols{ Cols },
        MatrixSize{ (unsigned long)Rows * (unsigned long)Cols }
{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
    MoveFromPointer(Init);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int NNSize, const NumType *Init, bool ByRow, ResourceManager *MM) :
        Vector<NumType>{ ByRow, MM }, Rows{ NNSize }, Cols{ NNSize },
        MatrixSize{ (unsigned long)Rows * (unsigned long)Cols }
{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
    MoveFromPointer(Init);

}

template<typename NumType>
Matrix1<NumType>::Matrix1(std::initializer_list<std::initializer_list<NumType>> Init, bool ByRow,
                          ResourceManager *MM) noexcept:
        Vector<NumType>(ByRow, MM)
{
    const std::initializer_list<NumType>* InitData = std::data(Init);

    if (ByRow) {
        Lines = Rows = (unsigned) Init.size();
        ElementsPerLine = Cols = (unsigned) (*InitData).size();
    }
    else {
        Lines = Cols = (unsigned) Init.size();
        ElementsPerLine = Rows = (unsigned) (*InitData).size();
    }

    MatrixSize = (unsigned long)Rows * (unsigned long)Cols;

    PerformSanityChecks();
    OptimizeResourceManagement();

    for (unsigned i = 0; i < Lines; ++i) {
        if (InitData[i].size() != ElementsPerLine)
            exit(0xfc);

        const NumType* InternalData = std::data(InitData[i]);
        for (unsigned j = 0; j < ElementsPerLine; ++j) {
            Array[i * SizeOfLine + j] = InternalData[j];
        }
    }
}

template<typename NumType>
Matrix1<NumType>::Matrix1(Matrix1 &&Target) noexcept :
        Vector<NumType>{std::move((Vector<NumType>&&)Target) }, Rows{ Target.Rows }, Cols{ Target.Cols } ,  MatrixSize{Target.MatrixSize },
        OffsetPerLine{ Target.OffsetPerLine },  SizeOfLine{ Target.SizeOfLine }, IsMemoryPacked{ Target.IsMemoryPacked }
{
    SetupAccess();
}

template<typename NumType>
Matrix1<NumType>::Matrix1(const Matrix1 &Target) noexcept :
        Vector<NumType>(Target), Rows{ Target.Rows }, Cols{ Target.Cols },
        MatrixSize{ Target.MatrixSize }, IsMemoryPacked{ Target.IsMemoryPacked },
        OffsetPerLine{ Target.OffsetPerLine }, SizeOfLine{ Target.SizeOfLine }
{
    SetupAccess();
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int Rows, unsigned int Cols, NumType InitVal, bool ByRow,
                          ResourceManager *MM) noexcept :
        Vector<NumType>{ ByRow, MM }, Rows{ Rows }, Cols{ Cols },
        MatrixSize { (unsigned long)Rows * (unsigned long)Cols }

{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement(&InitVal);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int NNSize, NumType InitVal, bool ByRow, ResourceManager *MM) noexcept :
        Vector<NumType>{ ByRow, MM }, Rows{ NNSize }, Cols{ NNSize },
        MatrixSize { (unsigned long)NNSize * (unsigned long)NNSize }

{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement(&InitVal);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int Rows, unsigned int Cols, bool ByRow, ResourceManager *MM) noexcept :
        Vector<NumType>{ ByRow, MM }, Rows{ Rows }, Cols{ Cols },
        MatrixSize { (unsigned long)Rows * (unsigned long)Cols }

{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
}

template<typename NumType>
Matrix1<NumType>::Matrix1(unsigned int NNSize, bool ByRow, ResourceManager *MM) noexcept :
        Vector<NumType>{ ByRow, MM }, Rows{ NNSize }, Cols{ NNSize },
        MatrixSize{ (unsigned long) NNSize * (unsigned long) NNSize}

{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
}

template<typename NumType>
void Matrix1<NumType>::MoveFromPointer(NumType *Src) {
    for (unsigned i = 0; i < Lines; ++i) {
        for (unsigned j = 0; j < ElementsPerLine; ++j)
            Array[i * SizeOfLine + j] = Src[i * ElementsPerLine + j];
    }
}

template<typename NumType>
void Matrix1<NumType>::SetupAccess()
// Copies corresponding dimensions to variables data alignment variables
{
    if (IsHorizontal) {
        ElementsPerLine = Cols;
        Lines = Rows;
        AccessFunc = &Matrix1::AccessByRow;
        AccessFuncConst = &Matrix1::AccessByRowConst;
    }
    else {
        ElementsPerLine = Rows;
        Lines = Cols;
        AccessFunc = &Matrix1::AccessByCol;
        AccessFuncConst = &Matrix1::AccessByColConst;
    }
}


#ifdef DEBUG_

template<typename NumType>
bool Matrix1<NumType>::CheckForIntegrity(NumType *Val, bool verbose)
// Passed data must be identically aligned to Array member data
{
    for (unsigned long i = 0; i < Lines; ++i)
        for (unsigned long j = 0; j < ElementsPerLine; ++j)
            if (Array[i * SizeOfLine + j] != Val[i * ElementsPerLine + j]) {
                if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: "
                                      << i << " and offset: " << j << std::endl;

                return false;
            }

    if (verbose) std::cout << "Success!!!\n";
    return true;
}

template<typename NumType>
bool Matrix1<NumType>::CheckForIntegrity(NumType Val, bool verbose) {
    for (unsigned long i = 0; i < Lines; ++i)
        for (unsigned long j = 0; j < ElementsPerLine; ++j)
            if (Array[i * SizeOfLine + j] != Val) {
                if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: "
                                      << i << " and offset: " << j << std::endl;
                return false;
            }

    if (verbose) std::cout << "Success!!!\n";
    return true;
}

#endif // DEBUG_

template<typename NumType>
Matrix1<NumType> &Matrix1<NumType>::operator=(const Matrix1 &x) {
    if (this == &x) return *this;

    Rows = x.Rows;
    Cols = x.Cols;
    OffsetPerLine = x.OffsetPerLine;
    IsMemoryPacked = x.IsMemoryPacked;
    MatrixSize = x.MatrixSize;
    SizeOfLine = x.SizeOfLine;

    *((Vector<NumType>*)this) = (const Vector<NumType>&)x;
    SetupAccess();

    return *this;
}

template<typename NumType>
Matrix1<NumType> &Matrix1<NumType>::operator=(Matrix1 &&x) noexcept {
    Vector<NumType>::DeallocateArray();

    Rows = x.Rows;
    Cols = x.Cols;
    IsMemoryPacked = x.IsMemoryPacked;
    OffsetPerLine = x.OffsetPerLine;
    Array = x.Array;
    Size = x.Size;
    IsHorizontal = x.IsHorizontal;
    SizeOfLine = x.SizeOfLine;

    x.Array = nullptr;

    SetupAccess();

    return *this;
}

template<typename NumType>
void Matrix1<NumType>::PrintWhole(std::ostream &out) {
    for (unsigned i = 0; i < Rows; ++i) {
        for (unsigned j = 0; j < Cols; ++j)
            out << std::setw(5) << operator()(i, j) << ' ';
        out << '\n';
    }
}

template<typename NumType>
void Matrix1<NumType>::PrintPartitioned(std::ostream &out, int MaxMatrixCols) {
    unsigned RowParts = Rows / 5;
    unsigned ColParts = Cols / MaxMatrixCols;
    unsigned i;

    auto PrintNRowsPartitioned = [&](unsigned StartingRow, unsigned RowCount) {
        unsigned j;
        static auto Helper = [&](unsigned Row, unsigned StartElement, unsigned Elements) {
            for (unsigned k = 0; k < Elements; ++k) {
                out << operator()(Row, StartElement + k) << ' ';
            }
            out << '\n';
        };

        for (j = 0; j < ColParts; ++j) {
            out << '\n' << "Rows in range=" << StartingRow << ':' << StartingRow + RowCount << " and cols in="
                << MaxMatrixCols * j << ':' << MaxMatrixCols * (j + 1) << "\n\n";
            for (unsigned k = 0; k < RowCount; ++k)
                Helper(StartingRow + k, MaxMatrixCols * j, MaxMatrixCols);
        }
        out << '\n' << "Rows in range=" << StartingRow << ':' << StartingRow + RowCount << " and cols in="
            << MaxMatrixCols * j << ':' << Cols << "\n\n";
        for (unsigned k = 0; k < RowCount; ++k)
            Helper(StartingRow + k, MaxMatrixCols * j, Cols - j * MaxMatrixCols);
    };

    for (i = 0; i < RowParts; ++i) {
        PrintNRowsPartitioned(5 * i, 5);
    }
    if (unsigned Rest = Rows - 5 * i) {
        PrintNRowsPartitioned(5 * i, Rest);
    }
}

template<typename NumType>
Matrix1<NumType> Matrix1<NumType>::GetTransposed() const
// Get a transposed copy of this matrix, actually naive slow algorithm
{
    Matrix1<NumType> RetVal(Cols, Rows, IsHorizontal, MM);

    TransposeMatrixRowStored(RetVal.Array, Array, Lines, ElementsPerLine, RetVal.SizeOfLine, SizeOfLine);

    return RetVal;
}

#endif