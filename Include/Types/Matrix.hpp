
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

// TODO OPISY
template<typename NumType>
class Matrix1;

template<typename NumType, size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
inline Matrix1<NumType> GetOuterProduct(const Vector<NumType>& A, const Vector<NumType>& B, bool HorizontalReturn = false);

template<typename NumT>
inline OuterProductMachine<NumT> GetOPM(const Vector<NumT>& A, const Vector<NumT>& B, Matrix1<NumT>& C);

template<typename NumT>
inline VMM<NumT> GetVMM(const Matrix1<NumT>& Mat, const Vector<NumT>& Vect, Vector<NumT>& RetVect);

template<typename NumT>
inline MatrixSumMachine<NumT> GetMSM(const Matrix1<NumT>& MatA, const Matrix1<NumT>& MatB, Matrix1<NumT>& MatC);

template<typename NumT>
inline GPMM<NumT> GetMultMachine(const Matrix1<NumT>& A, const Matrix1<NumT>& B, const Matrix1<NumT>& C);

template<typename NumType>
class Matrix1 : public Vector<NumType>
{
	//FixedSize - not expandable

	size_t Rows, Cols;
	// Contains only information about matrix size

	size_t ElementsPerLine{}, Lines{};
	// Contains only information about data storing and alignment

	size_t MatrixSize;
	// Store only data about mathematical matrix dimensions not actual on hw size

	size_t OffsetPerLine{};
	// Contains information how many data cells in memory are shifted to ensure data line alignment

	size_t SizeOfLine{};
	// Contains information about actual hw size of each line different from ElementsPerLine by alignment
	
	bool IsMemoryPacked = false; 
	// Mem optimizations for edge cases

	using Vector<NumType>::Array;
	using Vector<NumType>::IsHorizontal;
    using Vector<NumType>::ElementsPerCacheLine;
	using Vector<NumType>::MM;
	// Probably used in future to synchronize resource management

	using Vector<NumType>::Size;
	// actual on hw size expressed by number of held NumType variables inside the array

	using Vector<NumType>::CheckForIncorrectSize;
	NumType& (Matrix1::* AccessFunc)(size_t, size_t);
	const NumType& (Matrix1::* AccessFuncConst)(size_t, size_t) const;

	void AbandonIfVector() const {
		if (Rows == 1 || Cols == 1) [[unlikely]] {
            throw std::runtime_error("[ERROR] Passed sizes to matrix constructor describes vector\n");
		}
	}

	void PerformSanityChecks() const {
		AbandonIfVector();

		if (MatrixSize == 0) [[unlikely]]{
			throw std::runtime_error("[ERROR] One of passed matrix sizes was 0\n");
		}
	}

	void OptimizeResourceManagement();
	void SetupAccess();
	void MoveFromPointer(NumType *Src, size_t SrcSoL = 0);
public:
    // Basic constructors:
    Matrix1(size_t Rows, size_t Cols, bool ByRow = false, ResourceManager* MM = DefaultMM);
    Matrix1(std::initializer_list<std::initializer_list<NumType>> Init, bool ByRow = false, ResourceManager* MM = DefaultMM);
    Matrix1(size_t Rows, size_t Cols, const NumType* Init, bool ByRow = false, ResourceManager* MM = DefaultMM);
	Matrix1(const Matrix1& Target) noexcept;
	Matrix1(Matrix1&& Target) noexcept;

    // Actually delegates construction to one of the above ones
    Matrix1(size_t Rows, size_t Cols, NumType InitVal, bool ByRow = false, ResourceManager* MM = DefaultMM);
    Matrix1(size_t NNSize, bool ByRow = false, ResourceManager* MM = DefaultMM);
    Matrix1(size_t NNSize, NumType InitVal, bool ByRow = false, ResourceManager* MM = DefaultMM);
	Matrix1(size_t NNSize, const NumType* Init, bool ByRow = false, ResourceManager* MM = DefaultMM);

	// Data accessing operators

	inline NumType& AccessByRow(size_t Row, size_t Col) {
		return Array[Row * SizeOfLine + Col];
	}

	inline NumType& AccessByCol(size_t Row, size_t Col) {
		return Array[Col * SizeOfLine + Row];
	}

	inline const NumType& AccessByRowConst(size_t Row, size_t Col) const {
		return Array[Row * SizeOfLine + Col];
	}

	inline const NumType& AccessByColConst(size_t Row, size_t Col) const {
		return Array[Col * SizeOfLine + Row];
	}

	inline const NumType& operator()(size_t Row, size_t Col) const {
		return (this->*AccessFuncConst)(Row, Col);
	}

	inline NumType& operator()(size_t Row, size_t Col) {
		return (this->*AccessFunc)(Row, Col);
	}

	inline NumType& operator[](size_t Index) {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline const NumType& operator[](size_t Index) const {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline unsigned GetRows() const { return Rows; }
	inline unsigned GetCols() const { return Cols; }
	inline std::pair<size_t, size_t> GetDim() { return std::make_pair(Rows, Cols); }

#ifdef DEBUG_
	virtual bool CheckForIntegrity(NumType Val, bool verbose) const;
	virtual bool CheckForIntegrity(NumType* Val, bool verbose) const;
#endif // DEBUG_

	Matrix1& operator=(const Matrix1& x);
	Matrix1& operator=(Matrix1&& x) noexcept;

	friend std::ostream& operator<<(std::ostream& out, Matrix1& MyMatrix){
        size_t MaxMatrixCols = (FindConsoleWidth() - 2) / 6 > 0 ? (FindConsoleWidth() - 2) / 6 : 0;

        out << std::fixed << std::setprecision(3);

        if (MyMatrix.Cols <= MaxMatrixCols) {
            MyMatrix.PrintWhole(out);
        }
        else {
            MyMatrix.PrintPartitioned(out, MaxMatrixCols);
        }

        return out;
    }

private:
	void PrintWhole(std::ostream& out);
	void PrintPartitioned(std::ostream& out, size_t MaxMatrixCols);

public:
	// OPERATIONS ON MATRICES

	Matrix1 GetTransposed() const;

	const Matrix1& Transpose()
		// Uses naive algorithm
	{
		return *this = GetTransposed(*this);
	}

private:
    friend MatrixSumMachine<NumType> GetMSM<>(const Matrix1<NumType>& MatA, const Matrix1<NumType>& MatB, Matrix1<NumType>& MatC);
public:
    template<bool HorizontalReturn = false, size_t ThreadCap = 8, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
	friend Matrix1 operator+(const Matrix1& A, const Matrix1& B)
    {
        if (A.Rows != B.Rows || A.Cols != B.Cols) [[unlikely]]
            throw std::runtime_error("[ERROR] Not able to perform matrix sum, dimensions are not equal\n");
        Matrix1<NumType> RetVal{A.Rows, A.Cols};

        auto Machine = GetMSM(A, B, RetVal);
        Machine.template Perform<>();

        return RetVal;
    }

private:
    friend GPMM<NumType> GetMultMachine<>(const Matrix1<NumType> &A, const Matrix1<NumType> &B, const Matrix1<NumType> &C);
public:
    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
	friend Matrix1 operator*(const Matrix1& A, const Matrix1& B){
        if (A.Cols != B.Rows) [[unlikely]]
            throw std::runtime_error("[ERROR] Not able to perform matrix multiplication, wrong matrix sizes\n");

        Matrix1 RetVal(A.Rows, B.Cols, (NumType)0);
        GPMM<NumType> MultMachine = GetMultMachine(A,B,RetVal);
        MultMachine.template ExecuteOperation<ThreadCap, Decider>();

        return RetVal;
    }

private:
    friend VMM<NumType> GetVMM<>(const Matrix1<NumType>& Mat, const Vector<NumType>& Vect, Vector<NumType>& RetVect);

public:
    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
    friend Vector<NumType> operator*(const Matrix1& A, const Vector<NumType>& B){
        if (B.GetIsHorizontal() || A.Cols != B.GetSize())[[unlikely]]
            throw std::runtime_error("[ERROR] Not able to perform Vect and Matrix multiplication due to wrong sizes or dimensions\n");
        Vector<NumType> RetVal(A.Rows, (NumType)0, false);

        // TODO: Forward thredcap and decider
        auto Machine = GetVMM(A, B, RetVal);
        Machine.template PerformMV<ThreadCap, Decider>();

        return RetVal;
    }

    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LogarithmicThreads<ThreadCap>>
    friend Vector<NumType> operator*(const Vector<NumType>& A, const Matrix1& B){
        if (!A.GetIsHorizontal() || B.Rows != A.GetSize()) [[unlikely]]
            throw std::runtime_error("[ERROR] Not able to perform Vect and Matrix multiplication due to wrong sizes or dimensions\n");
        Vector<NumType> RetVal(B.Cols, (NumType)0, true);

        auto Machine = GetVMM(B, A, RetVal);
        Machine.template PerformVM<ThreadCap, Decider>();

        return RetVal;
    }

    Matrix1 GetModified(void (Vector<NumType>::*func)()) const{
        Matrix1 RetVal = *this;
        (RetVal.*func)();
        return RetVal;
    }

    Matrix1 GetModified(NumType(*func)(NumType x)) const{
        Matrix1 RetVal = *this;
        Vector<NumType>& Vect = RetVal;
//        Vect.ApplyOnDataEffect<func>(); TODO
        return RetVal;
    }

private:
    friend OuterProductMachine<NumType> GetOPM<>(const Vector<NumType>& A, const Vector<NumType>& B, Matrix1<NumType>& C);
public:
    template<typename NumT, size_t ThreadCap, size_t (*Decider)(size_t)>
    friend Matrix1<NumT> GetOuterProduct(const Vector<NumT>& A, const Vector<NumT>& B, bool HorizontalReturn);
};

// ------------------------------------
// Wrappers for different machines
// ------------------------------------

template<typename NumT>
inline MatrixSumMachine<NumT> GetMSM(const Matrix1<NumT>& MatA, const Matrix1<NumT>& MatB, Matrix1<NumT>& MatC){
    return MatrixSumMachine<NumT>(MatA.Array, MatB.Array, MatC.Array, MatA.Rows, MatA.Cols, MatA.Size, MatA.SizeOfLine,
                                  MatB.SizeOfLine, MatC.SizeOfLine, MatA.IsHorizontal, MatB.IsHorizontal, MatC.IsHorizontal);
}

template<typename NumT>
inline OuterProductMachine<NumT> GetOPM(const Vector<NumT>& A, const Vector<NumT>& B, Matrix1<NumT>& C)
{
    return OuterProductMachine<NumT>(A.GetArray(), B.GetArray(), C.Array, A.GetSize(), B.GetSize(), C.SizeOfLine, C.GetIsHorizontal());
}

template<typename NumT>
VMM<NumT> GetVMM(const Matrix1<NumT> &Mat, const Vector<NumT> &Vect, Vector<NumT> &RetVect) {
    return VMM<NumT>(Mat.Array, Vect.GetArray(), RetVect.GetArray(), Mat.Rows, Mat.Cols, Mat.SizeOfLine, Mat.IsHorizontal);
}

template<typename NumT>
GPMM<NumT> GetMultMachine(const Matrix1<NumT> &A, const Matrix1<NumT> &B, const Matrix1<NumT> &C) {
    return GPMM<NumT>(A.Array, B.Array, C.Array, A.Rows, A.Cols, B.Cols, A.SizeOfLine, B.SizeOfLine, C.SizeOfLine,
                      A.IsHorizontal, B.IsHorizontal, C.IsHorizontal);
}

// ------------------------------------
// Resource / Init Management
// ------------------------------------

template<typename NumType>
void Matrix1<NumType>::OptimizeResourceManagement()
// TODO: remove initval - perf
// Find optimal way to store the data, then prepares
// Arrays to be used efficiently, needs variable Rows and Cols to operate
{
    size_t ElementsOnLastCacheLine = ElementsPerLine % ElementsPerCacheLine;
    OffsetPerLine = ElementsOnLastCacheLine == 0 ? 0 : (ElementsPerCacheLine - ElementsOnLastCacheLine);
    SizeOfLine = ElementsPerLine + OffsetPerLine;
    size_t ExpectedAlignedSize = Lines * SizeOfLine;

#ifdef OPTIMISE_MEM_
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
#endif

    // Standard data storing with aligned every line of matrix
    *((Vector<NumType>*)this) = Vector<NumType>(ExpectedAlignedSize, IsHorizontal);
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

template<typename NumType>
Matrix1<NumType>::Matrix1(size_t Rows, size_t Cols, const NumType *Init, bool ByRow, ResourceManager *MM) :
        Vector<NumType>{ ByRow, MM }, Rows{ Rows }, Cols{ Cols },
        MatrixSize{ (unsigned long)Rows * (unsigned long)Cols }
{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
    MoveFromPointer(Init, 0);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(size_t NNSize, const NumType *Init, bool ByRow, ResourceManager *MM) :
    Matrix1(NNSize, NNSize, Init, ByRow, MM) {}

template<typename NumType>
Matrix1<NumType>::Matrix1(std::initializer_list<std::initializer_list<NumType>> Init, bool ByRow,
                          ResourceManager *MM) :
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

    for (size_t i = 0; i < Lines; ++i) {
        if (InitData[i].size() != ElementsPerLine) [[unlikely]]
            throw std::runtime_error("[ERROR] One of passed lines has different size than others\n");

        const NumType* InternalData = std::data(InitData[i]);
        for (size_t j = 0; j < ElementsPerLine; ++j) {
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
Matrix1<NumType>::Matrix1(size_t Rows, size_t Cols, NumType InitVal, bool ByRow, ResourceManager *MM)  :
          Matrix1(Rows, Cols, ByRow, MM)
{
    Vector<NumType>::SetWholeData(InitVal);
}

template<typename NumType>
Matrix1<NumType>::Matrix1(size_t NNSize, NumType InitVal, bool ByRow, ResourceManager *MM):
    Matrix1(NNSize, NNSize, InitVal, ByRow, MM) {}

template<typename NumType>
Matrix1<NumType>::Matrix1(size_t Rows, size_t Cols, bool ByRow, ResourceManager *MM)  :
        Vector<NumType>{ ByRow, MM }, Rows{ Rows }, Cols{ Cols },
        MatrixSize { (unsigned long)Rows * (unsigned long)Cols }

{
    PerformSanityChecks();
    SetupAccess();
    OptimizeResourceManagement();
}

template<typename NumType>
Matrix1<NumType>::Matrix1(size_t NNSize, bool ByRow, ResourceManager *MM)  :
    Matrix1(NNSize, NNSize, ByRow, MM){}

template<typename NumType>
void Matrix1<NumType>::MoveFromPointer(NumType * const Src, size_t SrcSoL)
    // Data from source has to be laid out in the same way as dst matrix is (by cols or by rows)
    // SrcSoL = 0 means there is no alignment done on it and function assumes its equal to matrix elements per line
{
    if (!SrcSoL) SrcSoL = ElementsPerLine;

    for (unsigned i = 0; i < Lines; ++i) {
        for (unsigned j = 0; j < ElementsPerLine; ++j)
            Array[i * SizeOfLine + j] = Src[i * SrcSoL + j];
    }
}

template<typename NumType>
Matrix1<NumType> &Matrix1<NumType>::operator=(const Matrix1 &x) {
    if (this == &x) [[unlikely]] return *this;

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

// ------------------------------------
// Debug Options
// ------------------------------------

#ifdef DEBUG_

template<typename NumType>
bool Matrix1<NumType>::CheckForIntegrity(NumType *Val, bool verbose) const
    // !!!Passed data must be identically aligned to Array member data
{
    for (size_t i = 0; i < Lines; ++i)
        for (size_t j = 0; j < ElementsPerLine; ++j)
            if (Array[i * SizeOfLine + j] != Val[i * ElementsPerLine + j])[[unlikely]] {
                if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: "
                                      << i << " and offset: " << j << std::endl;

                return false;
            }

    if (verbose) std::cout << "Success!!!\n";
    return true;
}

template<typename NumType>
bool Matrix1<NumType>::CheckForIntegrity(NumType Val, bool verbose) const {
    for (size_t i = 0; i < Lines; ++i)
        for (size_t j = 0; j < ElementsPerLine; ++j)
            if (Array[i * SizeOfLine + j] != Val) [[unlikely]] {
                if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: "
                                      << i << " and offset: " << j << std::endl;
                return false;
            }

    if (verbose) std::cout << "Success!!!\n";
    return true;
}

#endif // DEBUG_

// ------------------------------------
// Operations on object
// ------------------------------------

template<typename NumType>
void Matrix1<NumType>::PrintWhole(std::ostream &out) {
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j)
            out << std::setw(5) << operator()(i, j) << ' ';
        out << '\n';
    }
}

template<typename NumType>
void Matrix1<NumType>::PrintPartitioned(std::ostream &out, size_t MaxMatrixCols) {
    // TODO: Humanity level crime
    size_t RowParts = Rows / 5;
    size_t ColParts = Cols / MaxMatrixCols;
    size_t i;

    auto PrintNRowsPartitioned = [&](size_t StartingRow, size_t RowCount) {
        size_t j;
        static auto Helper = [&](size_t Row, size_t StartElement, size_t Elements) {
            for (size_t k = 0; k < Elements; ++k) {
                out << operator()(Row, StartElement + k) << ' ';
            }
            out << '\n';
        };

        for (j = 0; j < ColParts; ++j) {
            out << '\n' << "Rows in range=" << StartingRow << ':' << StartingRow + RowCount << " and cols in="
                << MaxMatrixCols * j << ':' << MaxMatrixCols * (j + 1) << "\n\n";
            for (size_t k = 0; k < RowCount; ++k)
                Helper(StartingRow + k, MaxMatrixCols * j, MaxMatrixCols);
        }
        out << '\n' << "Rows in range=" << StartingRow << ':' << StartingRow + RowCount << " and cols in="
            << MaxMatrixCols * j << ':' << Cols << "\n\n";
        for (size_t k = 0; k < RowCount; ++k)
            Helper(StartingRow + k, MaxMatrixCols * j, Cols - j * MaxMatrixCols);
    };

    for (i = 0; i < RowParts; ++i) {
        PrintNRowsPartitioned(5 * i, 5);
    }
    if (size_t Rest = Rows - 5 * i) {
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

template<typename NumT, size_t ThreadCap, size_t (*Decider)(size_t)>
Matrix1<NumT> GetOuterProduct(const Vector<NumT>& A, const Vector<NumT>& B, bool HorizontalReturn){
    if (A.GetIsHorizontal() || !B.GetIsHorizontal()){
        throw std::runtime_error("[ERROR] Dimensions of passed vector does not allow to perform OuterProduct\n");
    }

    Matrix1<NumT> RetVal(A.GetSize(), B.GetSize(), HorizontalReturn);
    auto Machine = GetOPM(A, B, RetVal);

    Machine.template Perform<ThreadCap, Decider>();
    return RetVal;
}

#endif