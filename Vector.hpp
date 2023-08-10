
// Author: Jakub Lisowski

#ifndef PARALLELNUM_VECTORS_H_
#define PARALLELNUM_VECTORS_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "NumericalCore.hpp"
#include "ErrorCodes.hpp"
#include "Debuggers.hpp"


template<typename T>
class Vector 
#ifdef DEBUG_
	: public DebuggerFoundation<T>
#endif
{
protected:
	//NumType ThreadedResult[MaxCPUThreads] = { NumType() };
	T* Array;
	bool IsHorizontal;
	const ResourceManager* MM;
	unsigned long Size;

	inline void SetWholeData(T Val)
		// EXTREMELY SLOW
	{
		if (Val == 0) {
#ifdef OpSysWIN_
			ZeroMemory(Array, Size * sizeof(T));
#else
			memset(Array, 0, Size * sizeof(NumType));
#endif
		}
		else {
			for (unsigned long i = 0; i < Size; ++i)
				Array[i] = Val;
		}
	}
	void CheckForIncorrectSize() const {
		if (Size == 0) {
			exit(0xf1);
		}
	}

	void AllocateArray() {
		if (MM) {
			//TODO
		}
		else {
			Array = (T*)_aligned_malloc(Size * sizeof(T), ALIGN);
			AbandonIfNull(Array);
		}
	}

	void DeallocateArray() {
		if (MM) {
			//TODO
		}
		else {
			_aligned_free(Array);
		}
	}

#ifdef DEBUG_
	virtual bool CheckForIntegrity(T Val) {
		for (unsigned long i = 0; i < Size; ++i)
			if (Array[i] != Val) return false;

		return true;
	}

	virtual bool CheckForIntegrity(T* Val) {
		for (unsigned long i = 0; i < Size; ++i)
			if (Array[i] != Val[i]) return false;

		return true;
	}
#endif

	// Used only for init_list<init_list> - unknow parameters
	// Should not be used as a class constructor, may lead to unexpected problems
	Vector(bool IsHorizontal, ResourceManager* MM) noexcept :
		IsHorizontal{ IsHorizontal }, MM{ MM }, Size{ 0 }, Array{ nullptr } {}

public:
	void MoveToArray(std::initializer_list<T> Init)
	{
		if (Init.size() != Size)
			exit(0xff);
		else if (Init.size() == 0)
			exit(0xfe);

		const T* Matrix = std::data(Init);

		for (unsigned long i = 0; i < Size; ++i)
			Array[i] = Matrix[i];
	}

	Vector(unsigned long Size, bool IsHorizontal = true, ResourceManager* MM = nullptr) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
	}

	Vector(unsigned long Size, T InitVal, bool IsHorizontal = true, ResourceManager* MM = nullptr) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
		SetWholeData(InitVal);
	}

	Vector(std::initializer_list<T> Init, bool IsHorizontal = true, ResourceManager* MM = nullptr) noexcept:
		Size{ Init.size() }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
		MoveToArray(Init);
	}

	Vector(const Vector& Target) noexcept:
		Size{ Target.Size }, IsHorizontal{ Target.IsHorizontal }, MM{ Target.MM }
	{
		AllocateArray();
		memcpy(Array, Target.Array, Target.Size * sizeof(T));
	}

	Vector(Vector&& Target) noexcept:
		Size{ Target.Size }, IsHorizontal{ Target.IsHorizontal }, MM{ Target.MM }, Array{ Target.Array }
	{
		Target.Array = nullptr;
	}

	Vector(unsigned long Size, T* Init, bool IsHorizontal = true, ResourceManager* MM = nullptr) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }, Array{ Init }
	{
        CheckForIncorrectSize();
		
		if (Init == nullptr)
			exit(0xfb);
	}

	Vector(unsigned long Size, const T* Init, bool IsHorizontal = true, ResourceManager* MM = nullptr) :
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();

		for (unsigned long i = 0; i < Size; ++i) {
			Array[i] = Init[i];
		}
	}

	Vector& operator=(const Vector& x) {
		if (this == &x) return *this;
		DeallocateArray();

		Size = x.Size;
		IsHorizontal = x.IsHorizontal;

		AllocateArray();
		memcpy(Array, x.Array, Size * sizeof(T));
		
		return *this;
	}

	Vector& operator=(Vector&& x) noexcept{
		DeallocateArray();

		Size = x.Size;
		IsHorizontal = x.IsHorizontal;
		Array = x.Array;

		x.Array = nullptr;
		return *this;
	}

	[[nodiscard]] inline unsigned long GetSize() const { return Size; }
	[[nodiscard]] inline bool GetHorizontalness() const { return IsHorizontal; }
	inline T* GetArray() const { return Array; }
	inline T* GetArray() { return Array; }
	inline void ChangePosition() { IsHorizontal = !IsHorizontal; }

	inline T& operator[](unsigned long x) { return Array[x]; }
	inline const T& operator[](unsigned long x) const { return Array[x]; }

	// Printing Vectors

private:
	void PrintHorizontally(std::ostream& out) const {
		unsigned long MaxPerCol = FindConsoleWidth() / 6;

		out << std::fixed << std::setprecision(3);
		
		for (unsigned long i = 0; i + MaxPerCol <= Size; i+=MaxPerCol) {
			out << "Vector values within index range: " << i << '-' << i + MaxPerCol
				<< ':' << std::endl;

			for (unsigned long j = 0; j < MaxPerCol; ++j) {
				out << Array[i+j] << ' ';
			}

			out << std::endl;
		}
	}

	void PrintVertically(std::ostream& out) const {
		out << std::endl;
		for (unsigned long i = 0; i < Size; ++i)
			out << Array[i] << '\n';
		out << std::endl;
	}

public:
	friend std::ostream& operator<<(std::ostream& out, Vector& Input) {
		if (Input.IsHorizontal)
			Input.PrintHorizontally(out);
		else
			Input.PrintVertically(out);

		return out;
	}

	// On-data operations

	void sqrt() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::sqrt(Array[i]);
	}

	void exp() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::exp(Array[i]);
	}

	void exp2() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::exp2(Array[i]);
	}

	void sin() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::sin(Array[i]);
	}

	void cos() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::cos(Array[i]);
	}

	void tan() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::tan(Array[i]);
	}

	void cot() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = 1 / std::tan(Array[i]);
	}

	void sinh() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::sinh(Array[i]);
	}

	void cosh() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::cosh(Array[i]);
	}

	void tanh() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = std::tanh(Array[i]);
	}

	void coth() {
		//#pragma omp for
		for (long i = 0; i < Size; ++i)
			Array[i] = 1 / std::tanh(Array[i]);
	}

	Vector GetModified(void (Vector::*func)()) const {
		Vector RetVal = *this;
		(RetVal.*func)();
		return RetVal;
	}

	Vector GetModified(T(*func)(T x)) const {
		Vector RetVal = *this;

		T* Dst = RetVal.Array;
		for (unsigned long i = 0; i < Size; ++i) {
			Dst[i] = func(Dst[i]);
		}

		return RetVal;
	}
private:
	template<class ThreadDecider = LogarithmicThreads, unsigned ThreadCap = 4>
	friend T DotProduct(const Vector& a, const Vector& b) {
		ThreadDecider Decider;
		T RetVal;
		const unsigned ThreadAmount = Decider(a.GetSize()) > ThreadCap ? ThreadCap : Decider(a.GetSize());

		if (ThreadAmount == 1) {
			RetVal = DotProduct(a.GetArray(), b.GetArray(), a.GetSize());
		}
		else {
			DotProductMachineChunked<T> Machine(a.GetArray(), b.GetArray(), ThreadAmount, a.GetSize());

			ThreadPackage& Threads = ResourceManager::GetThreads();
			for (unsigned i = 0; i < ThreadAmount; ++i) {
				Threads.Array[i] = new std::thread(&DotProductMachineChunked<T>::StartThread, &Machine, i);
			}

			for (unsigned i = 0; i < ThreadAmount; ++i) {
				Threads.Array[i]->join();
				delete Threads.Array[i];
			}

			RetVal = Machine.GetResult();
			Threads.Release();
		}

		return RetVal;
	}

public:
	// Vector and Vector operations
	friend T operator*(const Vector& a, const Vector& b) {
		if (a.GetHorizontalness() && !b.GetHorizontalness()) {
		
			if (a.GetSize() != b.GetSize()) {
#ifdef _MSC_VER 
				throw std::exception("Vectors are not the same length");
#else
				throw std::exception();
#endif
			}

			return DotProduct(a,b);
		}
		else {
#ifdef _MSC_VER 
			throw std::exception("Not matching dimension to perform dot product or they are not the same length");
#else
			throw std::exception();
#endif
		}
	}

};

#endif