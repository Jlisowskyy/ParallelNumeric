
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
#include "Debuggers.hpp"

ResourceManager* DefaultMatrixMM = nullptr;


template<typename T>
class Matrix1 : public Vector<T>
{
	//FixedSize - not expandable
	
	static const unsigned ElementsPerCacheLine = CACHE_LINE / sizeof(T);

	unsigned Rows, Cols; 
	// Contains only information about matrix size

	unsigned ElementsPerLine, Lines; 
	// Contains only information about data storing and alignment

	unsigned long MatrixSize; 
	// Store only data about mathematical matrix dimensions not actual on hw size

	unsigned OffsetPerLine;
	// Conatins hom many data cells memory is shifted to ensure data line alignment

	unsigned SizeOfLine; 
	// Contains information about actual hw size of each line diffrent than ElementsPerLine by alignment
	
	bool IsMemoryPacked = false; 
	// Memory optimizations for edge cases

	using Vector<T>::Array;
	using Vector<T>::IsHorizontal;

	using Vector<T>::MM; 
	// Probably used in future to synchronise resources management

	using Vector<T>::Size;
	// actual on hw size expressed by number of held T variables inside array

	using Vector<T>::CheckForIncorecctSize;
	T& (Matrix1::* AccesFunc)(const unsigned&, const unsigned&);
	const T& (Matrix1::* AccesFuncConst)(const unsigned&, const unsigned&) const;

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

	void OptimizeResourceManagment(T* InitVal = nullptr) 
		// Find optimal way to store the data, then prepares
		// Arrays to be used efficiently, needs variable Rows and Cols to operate
	{
		unsigned long PerCacheLine = CACHE_LINE / sizeof(T);
		unsigned long ElementsOnLastCacheLine = ElementsPerLine % PerCacheLine;
		OffsetPerLine = ElementsOnLastCacheLine == 0 ? 0 : (unsigned) (PerCacheLine - ElementsOnLastCacheLine);
		SizeOfLine = ElementsPerLine + OffsetPerLine;
		unsigned long ExpectedAlignedSize = Lines * SizeOfLine;
		unsigned long MemoryEnlargmentInBytes = OffsetPerLine * Lines * (unsigned long)sizeof(T);

		if (MemoryEnlargmentInBytes > (GB / 4) && (double)ExpectedAlignedSize > 1.2 * (double)MatrixSize )
			// Stores data partitioned to prevent cache lines overlapping
			// in case when alignemt could take more than 0.25Gb
			// Temporary barrier to be reconsidered in the future
		{
			IsMemoryPacked = true;
			OffsetPerLine = 0;
			SizeOfLine = ElementsPerLine;

			*((Vector<T>*)this) = InitVal == nullptr ?
				Vector<T>(MatrixSize, IsHorizontal) :
				Vector<T>(MatrixSize, *InitVal, IsHorizontal);
			
			return;
		}

		// Standard data storing with aligned every line of matrix
		*((Vector<T>*)this) = InitVal == nullptr ?
			Vector<T>(ExpectedAlignedSize, IsHorizontal) :
			Vector<T>(ExpectedAlignedSize, *InitVal, IsHorizontal);
	}

	void SetupAcces() 
		// Copies corresponding dimensions to variables data alignment variables
	{
		if (IsHorizontal) {
			ElementsPerLine = Cols;
			Lines = Rows;
			AccesFunc = &Matrix1::AccesByRow;
			AccesFuncConst = &Matrix1::AccesByRowConst;
		}
		else {
			ElementsPerLine = Rows;
			Lines = Cols;
			AccesFunc = &Matrix1::AccesByCol;
			AccesFuncConst = &Matrix1::AccesByColConst;
		}
	}

	void MoveFromPointer(T* Src) {
		const unsigned Range1 = Lines;
		const unsigned Range2 = ElementsPerLine;
		const unsigned SizeOfLine = this->SizeOfLine;

		for (unsigned i = 0; i < Range1; ++i) {
			for (unsigned j = 0; j < Range2; ++j)
				Array[i * SizeOfLine + j] = Src[i * ElementsPerLine + j];
		}
	}

public:
	Matrix1(unsigned NNSize, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) noexcept :
		Rows{ NNSize }, Cols{ NNSize }, MatrixSize{ (unsigned long) NNSize * (unsigned long) NNSize},
		Vector<T>{ ByRow, MM }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment();
	}

	Matrix1(unsigned Rows, unsigned Cols, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) noexcept :
		Rows{ Rows }, Cols{ Cols }, MatrixSize { (unsigned long)Rows * (unsigned long)Cols },
		Vector<T>{ ByRow, MM }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment();
	}

	Matrix1(unsigned NNSize, T InitVal, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) noexcept :
		Rows{ NNSize }, Cols{ NNSize }, MatrixSize { (unsigned long)NNSize * (unsigned long)NNSize },
		Vector<T>{ ByRow, MM }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment(&InitVal);
	}

	Matrix1(unsigned Rows, unsigned Cols, T InitVal, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) noexcept :
		Rows{ Rows }, Cols{ Cols }, MatrixSize { (unsigned long)Rows * (unsigned long)Cols },
		Vector<T>{ ByRow, MM }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment(&InitVal);
	}

	Matrix1(const Matrix1& Target) noexcept :
		Rows{ Target.Rows }, Cols{ Target.Cols }, MatrixSize{ Target.MatrixSize },
		IsMemoryPacked{ Target.IsMemoryPacked }, Vector<T>(Target),
		OffsetPerLine{ Target.OffsetPerLine }, SizeOfLine{ Target.SizeOfLine }
	{
		SetupAcces();
	}

	Matrix1(Matrix1&& Target) noexcept :
		Rows{ Target.Rows }, Cols{ Target.Cols } ,Vector<T>{ std::move((Vector<T>&&)Target) }, MatrixSize{ Target.MatrixSize },
		IsMemoryPacked{ Target.IsMemoryPacked }, OffsetPerLine{ Target.OffsetPerLine }, SizeOfLine{ Target.SizeOfLine }
	{
		SetupAcces();
	}

	Matrix1(std::initializer_list<std::initializer_list<T>> Init, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) noexcept:
		Vector<T>(ByRow, MM)
	{
		const std::initializer_list<T>* InitData = std::data(Init);

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
		OptimizeResourceManagment();

		const unsigned Range1 = Lines;
		const unsigned Range2 = ElementsPerLine;
		const unsigned SizeOfLine = this->SizeOfLine;

		for (unsigned i = 0; i < Range1; ++i) {
			if (InitData[i].size() != ElementsPerLine)
				exit(0xfc);

			const T* InternalData = std::data(InitData[i]);
			for (unsigned j = 0; j < Range2; ++j) {
				Array[i * SizeOfLine + j] = InternalData[j];
			}
		}
	}

	Matrix1(unsigned NNSize,const T* Init, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) :
		Rows{ NNSize }, Cols{ NNSize }, Vector<T>( ByRow, MM ),
		MatrixSize{ (unsigned long)Rows * (unsigned long)Cols }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment();
		MoveFromPointer(Init);

	}

	Matrix1(unsigned Rows, unsigned Cols,const T* Init, bool ByRow = false, ResourceManager* MM = DefaultMatrixMM) :
		Rows{ Rows }, Cols{ Cols }, Vector<T>( ByRow, MM ),
		MatrixSize{ (unsigned long)Rows * (unsigned long)Cols }
	{
		PerformSanityChecks();
		SetupAcces();
		OptimizeResourceManagment();
		MoveFromPointer(Init);
	}

	// Data accesing operators

	inline T& AccesByRow(const unsigned& Row, const unsigned& Col) {
		return Array[Row * SizeOfLine + Col];
	}

	inline T& AccesByCol(const unsigned& Row, const unsigned& Col) {
		return Array[Col * SizeOfLine + Row];
	}

	inline const T& AccesByRowConst(const unsigned& Row, const unsigned& Col) const {
		return Array[Row * SizeOfLine + Col];
	}

	inline const T& AccesByColConst(const unsigned& Row, const unsigned& Col) const {
		return Array[Col * SizeOfLine + Row];
	}

	inline const T& operator()(const unsigned& Row, const unsigned& Col) const {
		return (this->*AccesFuncConst)(Row, Col);
	}

	inline T& operator()(const unsigned& Row, const unsigned& Col) {
		return (this->*AccesFunc)(Row, Col);
	}

	inline T& operator[](const unsigned& Index) {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline const T& operator[](const unsigned& Index) const {
		return Array[Index + OffsetPerLine * (Index / ElementsPerLine)];
	}

	inline unsigned GetRows() { return Rows; }
	inline unsigned GetCols() { return Cols; }
	inline std::pair<unsigned, unsigned> GetDim() { return std::make_pair(Rows, Cols); }

#ifdef __DEBUG__
	virtual bool CheckForIntegrity(T Val, bool verbose = false) {
		for (unsigned long i = 0; i < Lines; ++i)
			for (unsigned long j = 0; j < ElementsPerLine; ++j)
				if (Array[i * SizeOfLine + j] != Val) {
					if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: " 
						<< i << " and offset: " << j << std::endl;
					return false;
				}

		if (verbose) std::cout << "Succes!!!\n";
		return true;
	}

	virtual bool CheckForIntegrity(T* Val, bool verbose = false) 
		// Passed data must be identically aligned to Array member data
	{
		for (unsigned long i = 0; i < Lines; ++i)
			for (unsigned long j = 0; j < ElementsPerLine; ++j)
				if (Array[i * SizeOfLine + j] != Val[i * ElementsPerLine + j]) {
					if (verbose)std::cerr << "[ERROR] Integrity test failed on Line: "
						<< i << " and offset: " << j << std::endl;

					return false;
				}

		if (verbose) std::cout << "Succes!!!\n";
		return true;
	}
#endif

	// Copying operators  

	const Matrix1& operator=(const Matrix1& x) {
		if (this == &x) return *this;

		Rows = x.Rows;
		Cols = x.Cols;
		OffsetPerLine = x.OffsetPerLine;
		IsMemoryPacked = x.IsMemoryPacked;
		MatrixSize = x.MatrixSize;
		SizeOfLine = x.SizeOfLine;

		*((Vector<T>*)this) = (const Vector<T>&)x;
		SetupAcces();
		
		return *this;
	}

	const Matrix1& operator=(Matrix1&& x) {
		Vector<T>::DeallocateArray();

		Rows = x.Rows;
		Cols = x.Cols;
		IsMemoryPacked = x.IsMemoryPacked;
		OffsetPerLine = x.OffsetPerLine;
		Array = x.Array;
		Size = x.Size;
		IsHorizontal = x.IsHorizontal;
		SizeOfLine = x.SizeOfLine;
		
		x.Array = nullptr;
		
		SetupAcces();

		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, Matrix1<T>& MyMatrix) {
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

	void PrintWhole(std::ostream& out) {
		for (unsigned i = 0; i < Rows; ++i) {
			for (unsigned j = 0; j < Cols; ++j)
				out << std::setw(5) << operator()(i, j) << ' ';
			out << '\n';
		}
	}

	void PrintPartitioned(std::ostream& out, int MaxMatrixCols) {
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
public:
	// OPERATIONS ON MATRICES

	Matrix1 GetTransposed() const
		// Get transposed copy of this matrix, actually naive slow algorythm
	{
		Matrix1 RetVal(Cols, Rows, IsHorizontal, MM);

		TransposeMatrixRowStored(RetVal.Array, Array, Lines, ElementsPerLine, RetVal.SizeOfLine, SizeOfLine);

		return RetVal;
	}

	const Matrix1& Transpose()
		// Uses naive algorytm
	{
		return *this = GetTransposed(*this);
	}

private:
	
	template<typename ThreadDeciderType>
	friend inline Matrix1 MatrixSumSameAcces(const Matrix1& a, const Matrix1& b, unsigned ThreadCap = 8) 
		// Handles the case when both matrices are stored in same way
	{
		Matrix1 RetVal(a.Rows, a.Cols, a.IsHorizontal);
		ThreadDeciderType ThreadDecider(ThreadCap);

		if (a.Size < ThreadDecider.StartingThreshold) {
			MatrixSumHelperAlignedArrays(RetVal.Array, a.Array, b.Array, a.Size);
			return RetVal;
		}

		unsigned ThreadAmount = ThreadDecider(a.Size);
		ThreadPackage Threads = ResourceManager::GetThreads();
		unsigned i;
		unsigned long ElementsPerThread = a.Size / (unsigned long)ThreadAmount;

		for (i = 0; i < ThreadAmount - 1; ++i) {
			unsigned long offset = i * ElementsPerThread;
			Threads.Array[i] = new std::thread(MatrixSumHelperAlignedArrays<T>,
				RetVal.Array + offset, a.Array + offset, b.Array + offset, ElementsPerThread
				);
		}

		unsigned long offset = i * ElementsPerThread;
		Threads.Array[i] = new std::thread(MatrixSumHelperAlignedArrays<T>,
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

	template<typename ThreadDeciderType>
	friend inline Matrix1 MatrixSumDiffAcces(const Matrix1& a, const Matrix1& b, unsigned ThreadCap) {
		Matrix1 RetVal(a.Rows, a.Cols, a.IsHorizontal);
		ThreadDeciderType ThreadDecider(ThreadCap);
		void (*Func)(T*, T*, T*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
		void (*FrameFunc)(T*, T*, T*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
		unsigned ElementsPerThread, DimToDivide, CoDim;

		if (!a.IsHorizontal && b.IsHorizontal) 
			// Left matrix is col stored and right is row stored
		{
			if (a.Rows > InequalityThreshold * a.Cols) 
				// Chosing preferable situatuation for block chunking
			{
				Func = &MatrixSumHelperNotAlignedArrays_CR_DivByRows<T>;
				FrameFunc = &MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame<T>;
				DimToDivide = a.Rows; CoDim = a.Cols;
			}
			else {
				Func = &MatrixSumHelperNotAlignedArrays_CR_DivByCols<T>;
				FrameFunc = &MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame<T>;
				DimToDivide = a.Cols; CoDim = a.Rows;
			}
		}
		else 
			// Opposite situation
		{
			if (a.Rows > InequalityThreshold * a.Cols) {
				Func = &MatrixSumHelperNotAlignedArrays_RC_DivByRows<T>;
				FrameFunc = &MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame<T>;
				DimToDivide = a.Rows; CoDim = a.Cols;
			}
			else {
				Func = &MatrixSumHelperNotAlignedArrays_RC_DivByCols<T>;
				FrameFunc = &MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame<T>;
				DimToDivide = a.Cols; CoDim = a.Rows;
			}
			
		}

		if (a.Size < ThreadDecider.StartingThreshold){
			ElementsPerThread = (DimToDivide / CACHE_LINE) * CACHE_LINE;

			Func(RetVal.Array, a.Array, b.Array, 0, ElementsPerThread, CoDim,
				RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine);
			FrameFunc(RetVal.Array, a.Array, b.Array, ElementsPerThread, DimToDivide, CoDim,
				RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine);
			
			return RetVal;
		}

		unsigned ThreadAmount = ThreadDecider(a.Size);
		ThreadPackage Threads = ResourceManager::GetThreads();
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

public:
	template<class TDT = LogarythmicThreads, unsigned ThreadCap = 8>
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
			std::cerr << "Not mathing Accesing types matrices - much slower operations\n\n";
		}

		// Solution A
		if (a.IsHorizontal == b.IsHorizontal )
			return MatrixSumSameAcces<TDT>(a, b, ThreadCap);
		else
			return MatrixSumDiffAcces<TDT>(a, b, ThreadCap);
	}
public:
	template<typename ThreadDecider = LogarythmicThreads, unsigned ThreadCap = 1>
	friend Matrix1 operator*(const Matrix1& a, const Matrix1& b)
	{
		if (a.Cols != b.Rows)
#ifdef _MSC_VER 
			throw std::exception("Not able to perform matrix multiplication - wrong matrix sizes");
#else
			throw std::exception();
#endif

		Matrix1 RetVal(a.Rows, b.Cols, (T)0);
		// TODO: look for entry level of optimisations to omit all optimisation checks
		ThreadDecider Decider(ThreadCap);
		unsigned long long OperationCount = RetVal.MatrixSize * a.Cols * b.Rows;

		unsigned ThreadAmount = Decider(OperationCount);
		unsigned TargetHorizontalBlocks = b.Cols / Matrix1<T>::ElementsPerCacheLine;
		unsigned BlocksPerThread = TargetHorizontalBlocks / ThreadAmount;

		//if (BlocksPerThread) {

		//}
		//else {
		//	unsigned TargetVerticalBlocks = a.Rows / Matrix1<T>::ElementsPerCacheLine;
		//	BlocksPerThread = TargetVerticalBlocks / ThreadAmount;

		//	if (BlocksPerThread) {

		//	}
		//	else {
		//		unsigned TranslationMatrixHorizontalBlocks = a.Cols / Matrix1<T>::ElementsPerCacheLine;
		//	}
		//}


		SimpleMultMachine<T> Machine(a.Rows, a.Cols, b.Rows, b.Cols, RetVal.SizeOfLine, a.SizeOfLine, b.SizeOfLine,
			RetVal.Array, a.Array, b.Array);

		//if (OperationCount < ThreadDecider.Threshold)
		if (true) {
			Machine.MultAlgo2_CC();
		}

		return RetVal;
	}

	Matrix1 GetModyfied(void (Vector<T>::* func)(void)) {
		Matrix1 RetVal = *this;
		RetVal.*func();
		return RetVal;
	}

	Matrix1 GetModyfied(T(*func)(T x)) {
		Matrix1 RetVal = *this;

		T* Dst = RetVal.Array;
		for (unsigned long i = 0; i < Size; ++i) {
			Dst[i] = func(Dst[i]);
		}

		return RetVal;
	}

private:
	template<typename T, class ThreadDecider = LogarythmicThreads, unsigned ThreadCap = 4>
	friend Matrix1<T> PerformOuterProduct(const Vector<T>& a, const Vector<T>& b);
public:
	template <typename T>
	friend Matrix1<T> OuterProduct(const Vector<T>& a, const Vector<T>& b);
};


template<typename T, class ThreadDecider, unsigned ThreadCap>
inline Matrix1<T> PerformOuterProduct(const Vector<T>& a, const Vector<T>& b)
{
	ThreadDecider Decider();
	Matrix1<T> RetVal(a.GetSize(), b.GetSize(), (T)0);
	const unsigned ThreadAmount = Decider(RetVal.GetSize()) > ThreadCap ? ThreadCap : Decider(RetVal.GetSize());

	OuterProductCol(RetVal.GetArray(), b.GetArray(), a.GetArray(), RetVal.GetDim());

	//if (ThreadAmount == 1) {
	//	result = OuterProduct(Result, b.GetArray(), a.GetArray());
	//}
	//else {
	//	//OuterProductMachineChunked<T> Machine(a.GetArray(), b.GetArray(), ThreadAmount, a.GetSize());

	//	//ThreadPackage& Threads = ResourceManager::GetThreads();
	//	//for (unsigned i = 0; i < ThreadAmount; ++i) {
	//	//	Threads.Array[i] = new std::thread(&DotProductMachineChunked<T>::StartThread, &Machine, i);
	//	//}

	//	//for (unsigned i = 0; i < ThreadAmount; ++i) {
	//	//	Threads.Array[i]->join();
	//	//	delete Threads.Array[i];
	//	//}

	//	//result = Machine.GetResult();
	//	//Threads.Release();
	//}

	return RetVal;
}

template<typename T>
inline Matrix1<T> OuterProduct(const Vector<T>& a, const Vector<T>& b)
{
	if (!a.GetHorizontalness() && b.GetHorizontalness()) {
		return PerformOuterProduct(a, b);
	}
	else {
#ifdef _MSC_VER 
		throw std::exception("Not matching dimension to perform outer product");
#else
		throw std::exception();
#endif
	}
}

#endif