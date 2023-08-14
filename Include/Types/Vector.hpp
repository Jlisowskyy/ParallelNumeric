
// Author: Jakub Lisowski

#ifndef PARALLELNUM_VECTORS_H_
#define PARALLELNUM_VECTORS_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "../Operations/NumericalCore.hpp"
#include "../Maintenance/ErrorCodes.hpp"
#include "../Maintenance/Debuggers.hpp"

extern ResourceManager* DefaultMM;

template<typename NumType>
class Vector;

template<typename NumType, unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LogarithmicThreads<ThreadCap>>
NumType DotProduct(const Vector<NumType>& a, const Vector<NumType>& b);

template<typename NumType>
class Vector 
#ifdef DEBUG_
	: public DebuggerFoundation<NumType>
#endif
{
protected:
	//NumType ThreadedResult[MaxCPUThreads] = { NumType() };
	NumType* Array;
	bool IsHorizontal;
	const ResourceManager* MM;
	unsigned long Size;

	inline void SetWholeData(NumType Val)
		// EXTREMELY SLOW
	{
		if (Val == 0) {
#ifdef OpSysWIN_
			ZeroMemory(Array, Size * sizeof(NumType));
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
			Array = (NumType*)_aligned_malloc(Size * sizeof(NumType), ALIGN);
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
	virtual bool CheckForIntegrity(NumType Val) {
		for (unsigned long i = 0; i < Size; ++i)
			if (Array[i] != Val) return false;

		return true;
	}

	virtual bool CheckForIntegrity(NumType* Val) {
		for (unsigned long i = 0; i < Size; ++i)
			if (Array[i] != Val[i]) return false;

		return true;
	}
#endif

	// Used only for init_list<init_list> - unknown parameters
	// Should not be used as a class constructor, may lead to unexpected problems
	Vector(bool IsHorizontal, ResourceManager* MM) noexcept :
		IsHorizontal{ IsHorizontal }, MM{ MM }, Size{ 0 }, Array{ nullptr } {}

public:
	void MoveToArray(std::initializer_list<NumType> Init);

	Vector(unsigned long Size, bool IsHorizontal = true, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
	}

	Vector(unsigned long Size, NumType InitVal, bool IsHorizontal = true, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
		SetWholeData(InitVal);
	}

	Vector(std::initializer_list<NumType> Init, bool IsHorizontal = true, ResourceManager* MM = DefaultMM) noexcept:
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
		memcpy(Array, Target.Array, Target.Size * sizeof(NumType));
	}

	Vector(Vector&& Target) noexcept:
		Size{ Target.Size }, IsHorizontal{ Target.IsHorizontal }, MM{ Target.MM }, Array{ Target.Array }
	{
		Target.Array = nullptr;
	}

	Vector(unsigned long Size, NumType* Init, bool IsHorizontal = true, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }, Array{ Init }
	{
        CheckForIncorrectSize();
        AbandonIfNull(Init);
	}

	Vector(unsigned long Size, const NumType* Init, bool IsHorizontal = true, ResourceManager* MM = DefaultMM) :
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();

		for (unsigned long i = 0; i < Size; ++i) {
			Array[i] = Init[i];
		}
	}

	Vector& operator=(const Vector& x);
	Vector& operator=(Vector&& x) noexcept;
	[[nodiscard]] inline unsigned long GetSize() const { return Size; }
	[[nodiscard]] inline bool GetHorizontalness() const { return IsHorizontal; }
	inline NumType* GetArray() const { return Array; }
	inline NumType* GetArray() { return Array; }
	inline void ChangePosition() { IsHorizontal = !IsHorizontal; }

	inline NumType& operator[](unsigned long x) { return Array[x]; }
	inline const NumType& operator[](unsigned long x) const { return Array[x]; }

	// Printing Vectors

private:
	void PrintHorizontally(std::ostream& out) const;
	void PrintVertically(std::ostream& out) const;

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

    void cot() {
        //#pragma omp for
        for (long i = 0; i < Size; ++i)
            Array[i] = 1 / std::tan(Array[i]);
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

	Vector GetModified(NumType(*func)(NumType x)) const {
		Vector RetVal = *this;

		NumType* Dst = RetVal.Array;
		for (unsigned long i = 0; i < Size; ++i) {
			Dst[i] = func(Dst[i]);
		}

		return RetVal;
	}
private:
    template<unsigned ThreadCap, unsigned (*Decider)(unsigned long long)>
	friend NumType DotProduct(const Vector& a, const Vector& b);

public:
	// Vector and Vector operations

    template<typename NumType2, unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LinearThreads<ThreadCap>>
    friend NumType2 operator*(const Vector<NumType2>& a, const Vector<NumType2>& b){
        if (a.GetHorizontalness() && !b.GetHorizontalness()) {

            if (a.GetSize() != b.GetSize()) {
#ifdef _MSC_VER
                throw std::exception("Vectors are not the same length");
#else
                throw std::exception();
#endif
            }

            return DotProduct<NumType, ThreadCap, Decider>(a,b);
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

template<typename NumType>
void Vector<NumType>::MoveToArray(std::initializer_list<NumType> Init) {
    if (Init.size() != Size)
        exit(0xff);
    else if (Init.size() == 0)
        exit(0xfe);

    const NumType* Matrix = std::data(Init);

    for (unsigned long i = 0; i < Size; ++i)
        Array[i] = Matrix[i];
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator=(Vector &&x) noexcept {
    DeallocateArray();

    Size = x.Size;
    IsHorizontal = x.IsHorizontal;
    Array = x.Array;

    x.Array = nullptr;
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator=(const Vector &x) {
    if (this == &x) return *this;
    DeallocateArray();

    Size = x.Size;
    IsHorizontal = x.IsHorizontal;

    AllocateArray();
    memcpy(Array, x.Array, Size * sizeof(NumType));

    return *this;
}

template<typename NumType>
void Vector<NumType>::PrintHorizontally(std::ostream &out) const {
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

template<typename NumType>
void Vector<NumType>::PrintVertically(std::ostream &out) const {
    out << std::endl;
    for (unsigned long i = 0; i < Size; ++i)
        out << Array[i] << '\n';
    out << std::endl;
}

template<typename NumType, unsigned ThreadCap, unsigned (*Decider)(unsigned long long)>
NumType DotProduct(const Vector<NumType> &a, const Vector<NumType> &b) {
    NumType RetVal;
    if (a.GetSize() < ThreadedStartingThreshold) {
        RetVal = DotProduct(a.GetArray(), b.GetArray(), a.GetSize());
    }
    else {
        const unsigned ThreadAmount = Decider(a.GetSize());
        DotProductMachineChunked<NumType> Machine(a.GetArray(), b.GetArray(), ThreadAmount, a.GetSize());

        ThreadPackage& Threads = ResourceManager::GetThreads();
        for (unsigned i = 0; i < ThreadAmount; ++i) {
            Threads.Array[i] = new std::thread(&DotProductMachineChunked<NumType>::StartThread, &Machine, i);
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

#endif