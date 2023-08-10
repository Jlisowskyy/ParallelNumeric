
// Author: Jakub Lisowski

#ifndef PARALLELNUM_HELPERS_H_
#define PARALLELNUM_HELPERS_H_

#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <latch>
#include <cstdlib>

#include "ResourceManager.hpp"
#include "NumericalCoreMatrixMultiplication.hpp"

// ------------------------------------
// Thread Amount management

class LogarythmicThreads
	// Function to decide about amount of needed threads
{
	const unsigned ThreadCap;
public:
	static const long long unsigned StartingThreshold = 32768;

	LogarythmicThreads(unsigned ThreadCap = MaxCPUThreads) : ThreadCap{ ThreadCap } {}
	inline unsigned operator()(unsigned long long Elements) const {
		unsigned Ret = (unsigned)(log2(Elements / StartingThreshold)) + 1;
		return Ret < ThreadCap ? Ret : ThreadCap;
	}
};

class LinearThreads
	// Same as upper one
{
	const unsigned ThreadCap;
public:
	static const long long unsigned StartingThreshold = 32768;

	LinearThreads(unsigned ThreadCap = MaxCPUThreads) : ThreadCap{ ThreadCap } {}
	unsigned operator()(unsigned long long Elements) const {
		unsigned Ret = (unsigned)(Elements / StartingThreshold);
		return Ret < ThreadCap ? Ret : ThreadCap;
	}
};

// ------------------------------------------
// Sum of matrices functions

template<typename T>
void MatrixSumHelperAlignedArrays(T* const Target, const T* const Input1, const T* const Input2, const unsigned long Elements)
// Function used to Sum matrices
{
	for (unsigned long i = 0; i < Elements; ++i)
		Target[i] = Input1[i] + Input2[i];
}

#ifdef __AVX__

template<>
void MatrixSumHelperAlignedArrays(double* Target,const double* const Input1, const double* const Input2, const unsigned long Elements) {
	const __m256d* const VectInput1 = (const __m256d* const)Input1;
	const __m256d* const VectInput2 = (const __m256d* const)Input2;
	__m256d* VectTarget = (__m256d*)Target;

	const unsigned long VectSize = Elements / DOUBLE_VECTOR_LENGTH;
	for (unsigned long i = 0; i < VectSize; ++i) {
		VectTarget[i] = _mm256_add_pd(VectInput1[i], VectInput2[i]);
	}

	for (unsigned long i = VectSize * DOUBLE_VECTOR_LENGTH; i < Elements; ++i) {
		Target[i] = Input1[i] + Input2[i];
	}
}

template<>
void MatrixSumHelperAlignedArrays(float* Target, const float* const Input1, const float* const Input2, const unsigned long Elements) {
	const __m256* const VectInput1 = (const __m256* const)Input1;
	const __m256* const VectInput2 = (const __m256* const)Input2;
	__m256* VectTarget = (__m256*)Target;

	const unsigned long VectSize = Elements / FLOAT_VECTOR_LENGTH;
	for (unsigned long i = 0; i < VectSize; ++i) {
		VectTarget[i] = _mm256_add_ps(VectInput1[i], VectInput2[i]);
	}

	for (unsigned long i = VectSize * FLOAT_VECTOR_LENGTH; i < Elements; ++i) {
		Target[i] = Input1[i] + Input2[i];
	}
}

#endif

template<typename T>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartCol, const unsigned StopCol, const unsigned Rows,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine) 
	// Function assumes that passed Start and StopCol ale divisible by T corresponding length of cache line
	// otherwise behavior is undefined. Also all data pointer should be aligned to cache lines otherwise 
	// operation may be much slower.
{
	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartRow, const unsigned StopRow, const unsigned Cols,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine)
	// Function assumes that passed Start and StopCol ale divisible by T corresponding length of cache line
	// otherwise behavior is undefined. Also all data pointer should be aligned to cache lines otherwise 
	// operation may be much slower.
{
	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartCol, const unsigned StopCol, const unsigned Rows,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine) 
	// Function assumes that passed Start and StopCol ale divisible by T corresponding length of cache line
	// otherwise behavior is undefined. Also all data pointer should be aligned to cache lines otherwise 
	// operation may be much slower.
{
	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartRow, const unsigned StopRow, const unsigned Cols,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine)
	// Function assumes that passed Start and StopCol ale divisible by T corresponding length of cache line
	// otherwise behavior is undefined. Also all data pointer should be aligned to cache lines otherwise 
	// operation may be much slower.
{
	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_CR_DivByCols_Frame(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartCol, const unsigned StopCol, const unsigned Rows,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine) 
{
	if (StartCol == StopCol) return;

	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_CR_DivByRows_Frame(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartRow, const unsigned StopRow, const unsigned Cols,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine)
{
	if (StartRow == StopRow) return;

	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_RC_DivByRows_Frame(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartRow, const unsigned StopRow, const unsigned Cols,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine)
{
	if (StartRow == StopRow) return;

	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

template<typename T>
void MatrixSumHelperNotAlignedArrays_RC_DivByCols_Frame(T* Target, const T* const Input1, const T* const Input2,
	const unsigned StartCol, const unsigned StopCol, const unsigned Rows,
	const unsigned TargetSizeOfLine, const unsigned Input1SizeOfLine, const unsigned Input2SizeOfLine)
{
	if (StartCol == StopCol) return;

	const unsigned ElementsInCacheLine = CACHE_LINE / sizeof(T);
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

//#define I2A(offset) Input2[(i + z + offset) * Input2SizeOfLine + (j + k)]
//
//template<>
//void MatrixSumHelperNotAlignedArrays_CR(double* Target, double* Input1, double* Input2, unsigned StartCol, unsigned StopCol, unsigned Rows,
//	unsigned TargetSizeOfLine, unsigned Input1SizeOfLine, unsigned Input2SizeOfLine) {
//
//	unsigned ElementsInCacheLine = CACHE_LINE / sizeof(double);
//	unsigned VectorsInCacheLine = 2;
//
//	for (unsigned i = 0; i < Rows; i += ElementsInCacheLine) {
//		for (unsigned j = StartCol; j < StopCol; j += ElementsInCacheLine) {
//			for (unsigned k = 0; k < ElementsInCacheLine; ++k) {
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





// -------------------------------------
// Matrix transposition solutions

// Naive solution 
template<typename T>
void TransposeMatrixRowStored(T* Dst, T* Src, const unsigned SrcLines, const unsigned SrcElementsPerLine, 
	const unsigned DstSizeOfLine, const unsigned SrcSizeOfLine) {
	for (unsigned i = 0; i < SrcLines; ++i) {
		for (unsigned j = 0; j < SrcElementsPerLine; ++j) {
			Dst[j * DstSizeOfLine + i] = Src[i * SrcSizeOfLine + j];
		}
	}
}

// ---------------------------------
// Dot product code 

template<typename T>
T DotProduct(T* const Src1, T* const Src2, unsigned long Range) {
	T result = T();

	for (unsigned long i = 0; i < Range; i++) {
		result += Src1[i] * Src2[i];
	}

	return result;
}

template<>
double DotProduct(double* const Src1, double* const Src2, unsigned long Range) {
	__m256d* const VectSrc1 = (__m256d* const) Src1;
	__m256d* const VectSrc2 = (__m256d* const) Src2;
	__m256d Store = _mm256_set_pd(0, 0, 0, 0);

	const unsigned long VectRange = Range/4;
	for (unsigned long i = 0; i < VectRange; ++i) {
		Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
	}

	double EndResult = 0;
	for (unsigned long i = VectRange * 4; i < Range; ++i) {
		EndResult += Src1[i] * Src2[i];
	}

	double* result =(double*) &Store;
	return result[0] + result[1] + result[2] + result[3] + EndResult;
}

// Solution 1

template<typename T>
class DPMCore{
protected:
	const T* const Src1;
	const T* const Src2;
	const unsigned Threads;
	const unsigned long Range;
	const unsigned long EndIndex;
	T ResultArray[MaxCPUThreads] = { T() };
	std::latch Counter;
	std::latch WriteCounter;
public:
	DPMCore(const T* const Src1, const T* const Src2, unsigned Threads, unsigned long Range, unsigned long EndIndex) :
		EndIndex{ EndIndex }, Src1{ Src1 }, Src2{ Src2 }, Threads{ Threads }, Range{ Range },
		Counter{ Threads }, WriteCounter{ Threads }
	{}

	T GetResult() {
		T Ret = T();

		for (unsigned long i = EndIndex; i < Range; ++i) {
			Ret += Src1[i] * Src2[i];
		}

		for (unsigned i = 0; i < Threads; ++i) {
			Ret += ResultArray[i];
		}

		return Ret;
	}
};

// Solution 2

template<typename T>
class DotProductMachineChunked: public DPMCore<T> {
	const unsigned long ElemPerThread;
public:
	DotProductMachineChunked(const T* const Src1, const T* const Src2, unsigned Threads, unsigned long Range) :
		ElemPerThread{ Range / (unsigned long) Threads }, DPMCore<T>(Src1, Src2, Threads, Range, (Range / (unsigned long)Threads) * (unsigned long)Threads)
	{}

	void StartThread(unsigned ThreadID) {
		T Ret = T();
		const T* const S1 = DPMCore<T>::Src1;
		const T* const S2 = DPMCore<T>::Src2;
		const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;

		DPMCore<T>::Counter.arrive_and_wait();
		for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
			Ret += S1[i] * S2[i];
		}

		DPMCore<T>::WriteCounter.arrive_and_wait();
		DPMCore<T>::ResultArray[ThreadID] = Ret;
	}
};

// Double spec
template<>
DotProductMachineChunked<double>::DotProductMachineChunked(const double* const Src1, const double* const Src2, unsigned Threads, unsigned long Range) :
	ElemPerThread{ Range / (Threads * DOUBLE_VECTOR_LENGTH) }, 
	DPMCore<double>(Src1, Src2, Threads, Range, (Range / (Threads * DOUBLE_VECTOR_LENGTH)) * Threads * DOUBLE_VECTOR_LENGTH)
{}

template<>
void DotProductMachineChunked<double>::StartThread(unsigned ThreadID) {
	const __m256d* const VectSrc1 = (const __m256d* const) Src1;
	const __m256d* const VectSrc2 = (const __m256d* const) Src2;
	__m256d Store = _mm256_set_pd(0, 0, 0, 0);
	const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;

	Counter.arrive_and_wait();
	for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
		Store = _mm256_fmadd_pd(VectSrc1[i], VectSrc2[i], Store);
	}

	WriteCounter.arrive_and_wait();
	double* Result = (double*) &Store;
	ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3];
}


//// Float spec
template<>
DotProductMachineChunked<float>::DotProductMachineChunked(const float* const Src1, const float* const Src2, unsigned Threads, unsigned long Range) :
	ElemPerThread{ Range / (Threads * FLOAT_VECTOR_LENGTH) }, 
	DPMCore<float>(Src1, Src2, Threads, Range, (Range / (Threads * FLOAT_VECTOR_LENGTH)) * Threads * FLOAT_VECTOR_LENGTH)
{}

template<>
void DotProductMachineChunked<float>::StartThread(unsigned ThreadID) {
	Counter.arrive_and_wait();
	__m256* const VectSrc1 = (__m256* const) Src1;
	__m256* const VectSrc2 = (__m256* const) Src2;
	__m256 Store = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

	const unsigned long LoopRange = (ThreadID + 1) * ElemPerThread;
	for (unsigned long i = ThreadID * ElemPerThread; i < LoopRange; ++i) {
		Store = _mm256_fmadd_ps(VectSrc1[i], VectSrc2[i], Store);
	}

	WriteCounter.arrive_and_wait();
	float* Result = (float*)&Store;
	ResultArray[ThreadID] = Result[0] + Result[1] + Result[2] + Result[3] + Result[4] + Result[5] + Result[6] + Result[7];
}

#define SingleOPStraight(offset) TempArray[offset] += S1[i + offset] * S2[i + offset]

template<typename T>
class DotProductMachineComb: public DPMCore<T> {
	const unsigned long LoopRange;
	const unsigned long PerCircle = CACHE_LINE / sizeof(T);
public:
	DotProductMachineComb(const T* const Src1, const T* const Src2, unsigned Threads, unsigned long Range) :
		LoopRange{ Range }, DPMCore<T>(Src1, Src2, Threads, Range, (Range / (CACHE_LINE / sizeof(T))) * (CACHE_LINE / sizeof(T)))
	{}

	void StartThread(unsigned ThreadID) {
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
};

// Double Spec
#define SingleOP_D(offset) Store = _mm256_fmadd_pd(VectSrc1[i+offset], VectSrc2[i+offset], Store)
#define PER_CIRCLE_DOUBLE 2

template<>
DotProductMachineComb<double>::DotProductMachineComb(const double* const Src1, const double* const Src2, unsigned Threads, unsigned long Range) :
	PerCircle{ PER_CIRCLE_DOUBLE }, LoopRange{ Range / DOUBLE_VECTOR_LENGTH }, 
	DPMCore<double>(Src1, Src2, Threads, Range, 
		(((Range / DOUBLE_VECTOR_LENGTH) * DOUBLE_VECTOR_LENGTH) / PER_CIRCLE_DOUBLE) * (CACHE_LINE / PER_CIRCLE_DOUBLE))
{}	

template<>
void DotProductMachineComb<double>::StartThread(unsigned ThreadID)
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
	double* result = (double*)&Store;

	for (int i = 0; i < 4; ++i) {
		ret += result[i];
	}

	ResultArray[ThreadID] = ret;
}

// Float Spec
#define SingleOP_F(offset) Store =_mm256_fmadd_ps(VectSrc1[i+offset], VectSrc2[i+offset], Store)
#define PER_CIRCLE_FLOAT 2

template<>
DotProductMachineComb<float>::DotProductMachineComb(const float* const Src1, const float* const Src2, unsigned Threads, unsigned long Range) :
	PerCircle{ PER_CIRCLE_FLOAT }, LoopRange{ Range / FLOAT_VECTOR_LENGTH },
	DPMCore<float>(Src1, Src2, Threads, Range, (((Range / FLOAT_VECTOR_LENGTH)* FLOAT_VECTOR_LENGTH) / PER_CIRCLE_FLOAT)* PER_CIRCLE_FLOAT)
{}

template<>
void DotProductMachineComb<float>::StartThread(unsigned ThreadID)
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
	float* RetP = (float*)&Store;

	for (int i = 0; i < 4; ++i) {
		RetVal += RetP[i];
	}

	ResultArray[ThreadID] = RetVal;
}


// Outer Product

template<typename T>
void OuterProductCol(T* Dst, const T* const Src1, const T* const Src2, std::pair<unsigned, unsigned> Dim) {
	for (unsigned i = 0; i < Dim.second; ++i) {
		for (unsigned j = 0; j < Dim.first; ++j) {
			Dst[i * Dim.first + j] = Src1[j] * Src2[i];
		}
	}
}


#define pt (const __m256d* const)

#define LOB(offset) VectDst[offset][0] = _mm256_mul_pd(*(pt(Src1 + j)), Mult[offset]); VectDst[offset][1] = _mm256_mul_pd(*(pt(Src1 + j + 4)), Mult[offset]);

#define LoadDst(offset) (__m256d*) (Dst + (i + offset) * Dim.first);

template<>
void OuterProductCol(double* Dst, const double* const Src1, const double* const Src2, std::pair<unsigned, unsigned> Dim) {
	//const unsigned long BlockSize = 8;
	//const unsigned long BlockedRangeHorizontal = 
	//	Dim.second >= BlockSize ? Dim.second - BlockSize : 0; // CHECK
	//const unsigned long BlockedRangeVertical = 
	//	Dim.first >= BlockSize ? Dim.first - BlockSize : 0; // CHECK

	//for (unsigned long i = 0; i < BlockedRangeHorizontal; i += BlockSize) {
	//	for (unsigned long j = 0; j < BlockedRangeVertical; j += BlockSize) {
	//		for (unsigned long k = i; k < i + BlockSize; ++k) {
	//			for (unsigned long z = j; z < j + BlockSize; ++z) {
	//				Dst[k * Dim.second + z] = Src1[k] * Src2[z];
	//			}
	//		}
	//	}
	//}

	const unsigned long BlockSize = 8;
	const unsigned long BlockedRangeHorizontal =
		Dim.second >= BlockSize ? Dim.second - BlockSize : 0; // CHECK
	const unsigned long BlockedRangeVertical =
		Dim.first >= BlockSize ? Dim.first - BlockSize : 0; // CHECK

	const __m256d* const S1 = (const __m256d* const) (Src1);
	__m256d** VectDst = (__m256d**) _aligned_malloc(sizeof(__m256d*) * BlockSize, ALLIGN);
	__m256d* Mult = (__m256d*) _aligned_malloc(sizeof(__m256d) * BlockSize, ALLIGN);

	if (!VectDst || !Mult) {
		std::cout << "alloc err\n";
		exit(0);
	}
	
	for (unsigned long i = 0; i < BlockedRangeHorizontal; i += BlockSize) {
		
		for (int z = 0; z < BlockSize; ++z) {
			VectDst[z] = LoadDst(z);
			Mult[z] = _mm256_set1_pd(Src2[i + z]);
		}

		for (unsigned long j = 0; j < BlockedRangeVertical; j += BlockSize) {
			for (int z = 0; z < BlockSize; ++z) {
				LOB(z);
			}
		}

		_mm256_add_epi64(((__m256i*)VectDst)[0], _mm256_set1_epi64x(2));
		_mm256_add_epi64(((__m256i*)VectDst)[1], _mm256_set1_epi64x(2));
	}

	_aligned_free(Mult);
	_aligned_free(VectDst);
}


#endif