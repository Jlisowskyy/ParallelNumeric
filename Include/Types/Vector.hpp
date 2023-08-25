
// Author: Jakub Lisowski

#ifndef PARALLELNUM_VECTORS_H_
#define PARALLELNUM_VECTORS_H_

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <functional>

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
    static constexpr unsigned ElementsPerCacheLine = CACHE_LINE / sizeof(NumType);
    unsigned long Size;
	bool IsHorizontal;
	const ResourceManager* MM;
    NumType* Array;

	inline void SetWholeData(NumType Val);
    void CheckForIncorrectSize() const { if (Size == 0) exit(0xf1); }
	void AllocateArray();
	void DeallocateArray();

#ifdef DEBUG_
	virtual bool CheckForIntegrity(NumType Val, bool verbose);
	virtual bool CheckForIntegrity(NumType* Val, bool verbose);
#endif

	// Used only for init_list<init_list> - unknown parameters
	// Should not be used as a class constructor, may lead to unexpected problems
	Vector(bool IsHorizontal, ResourceManager* MM) noexcept :
        Size{ 0 }, IsHorizontal{ IsHorizontal }, MM{ MM }, Array{ nullptr } {}

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

    ~Vector(){
        DeallocateArray();
    }

	Vector& operator=(const Vector& x);
	Vector& operator=(Vector&& x) noexcept;
    inline unsigned long GetSize() const { return Size; }
	inline bool GetIsHorizontal() const { return IsHorizontal; }
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

    template<NumType(*UnaryOperation)(NumType)>
    inline void ApplyOnDataEffect()
        // Transforms data using templated function
    {
        #pragma omp parallel for
        for (size_t i = 0; i < Size; ++i) {
            Array[i] = UnaryOperation(Array[i]);
        }
    }

#if defined(__AVX__) || defined(__AVX2__)
    template<typename AVXType, AVXType (AVXOperation)(AVXType), NumType(*UnCleaningOperation)(NumType)>
    inline void ApplyAVXOnDataEffect()
        // Transforms data with avx function
    {
        static constexpr size_t PackageSize = AVX_SIZE / sizeof(NumType);
        const size_t Range = (Size / ElementsPerCacheLine) * ElementsPerCacheLine;

        #pragma omp parallel for
        for (size_t i = 0; i < Range; i+= ElementsPerCacheLine) {
            *((AVXType*)(Array + i)) = AVXOperation(*((AVXType*)(Array + i)));
            *((AVXType*)(Array + i + PackageSize)) = AVXOperation(*((AVXType*)(Array + i + PackageSize)));
        }
        for(size_t i = Range; i < Size; ++i){
            Array[i] = UnCleaningOperation(Array[i]);
        }
    }
#endif // __AVX__ __AVX2__


	// On-data operations
	void sqrt() {
        ApplyOnDataEffect<std::sqrt>();
	}

    void reciprocal(){
        auto operand = [](NumType x) -> NumType{ return 1 / x; };
        ApplyOnDataEffect<operand>();
    }

    void rsqrt(){
        auto operand = [](NumType x) -> NumType{ return 1 / std::sqrt(x); };
        ApplyOnDataEffect<operand>();
    }

	void exp() {
        ApplyOnDataEffect<std::exp>();
	}

	void exp2() {
        ApplyOnDataEffect<std::exp2>();
	}

	void sin() {
        ApplyOnDataEffect<std::sin>();
	}

	void cos() {
        ApplyOnDataEffect<std::cos>();
	}

	void tan() {
        ApplyOnDataEffect<std::tan>();
	}

	void sinh() {
        ApplyOnDataEffect<std::sinh>();
	}

	void cosh() {
        ApplyOnDataEffect<std::cosh>();
	}

	void tanh() {
        ApplyOnDataEffect<std::tanh>();
	}

    void cot() {
        auto operand = [](NumType x) -> NumType { return 1 / std::tan(x); };
        ApplyOnDataEffect<operand>();
    }

	void coth() {
        auto operand = [](NumType x) -> NumType { return 1 / std::tanh(x); };
        ApplyOnDataEffect<std::exp>();
	}

	Vector GetModified(void (Vector::*func)()) const {
		Vector RetVal = *this;
		(RetVal.*func)();
		return RetVal;
	}

	Vector GetModified(NumType(*func)(NumType x)) const {
		Vector RetVal = *this;
        ApplyOnDataEffect<func>();

		return RetVal;
	}
private:
    template<unsigned ThreadCap, unsigned (*Decider)(unsigned long long)>
	friend NumType DotProduct(const Vector& a, const Vector& b);

public:
	// Vector and Vector operations

    template<unsigned ThreadCap = 8, unsigned (*Decider)(unsigned long long) = LinearThreads<ThreadCap>>
    friend NumType operator*(const Vector<NumType>& a, const Vector<NumType>& b){
        if (a.GetIsHorizontal() && !b.GetIsHorizontal()) {

            if (a.GetSize() != b.GetSize()) {
                throw std::runtime_error("Vectors are not the same length\n");
            }

            return DotProduct<NumType, ThreadCap, Decider>(a,b);
        }
        else {
            throw std::runtime_error("Not matching dimension to perform dot product or they are not the same length\n");
        }
    }

};

//-----------------------------------------
// High perf AVX spec
//-----------------------------------------

#ifdef __AVX__

template<>
void Vector<double>::sqrt();
template<>
void Vector<float>::sqrt();
template<>
void Vector<float>::reciprocal();

#endif // __AVX__

//-----------------------------------------
// Template implementation
//-----------------------------------------

#ifdef DEBUG_

template<typename NumType>
bool Vector<NumType>::CheckForIntegrity(NumType *Val, bool verbose) {
    for (unsigned long i = 0; i < Size; ++i)
        if (Array[i] != Val[i]){
            if (verbose) std::cerr << "[ERROR] Integrity test failed on Index: " << i << '\n';
            return false;
        }

    if (verbose) std::cout << "Success\n";
    return true;
}

template<typename NumType>
bool Vector<NumType>::CheckForIntegrity(NumType Val, bool verbose) {
    for (unsigned long i = 0; i < Size; ++i)
        if (Array[i] != Val) {
            if (verbose) std::cerr << "[ERROR] Integrity test failed on Index: " << i << '\n';
            return false;
        }

    if (verbose) std::cout << "Success\n";
    return true;
}

#endif // DEBUG_

template<typename NumType>
void Vector<NumType>::SetWholeData(NumType Val)
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

template<typename NumType>
void Vector<NumType>::DeallocateArray() {
    if (MM) {
        //TODO
    }
    else {
#ifdef OpSysWIN_
        _aligned_free(Array);
#elif defined OpSysUNIX_
        free(Array);
#endif
    }
}

template<typename NumType>
void Vector<NumType>::AllocateArray() {
    if (MM) {
        //TODO
    }
    else {
        short tries = 5;
        Array = nullptr; // <------- Temp
        while(Array == nullptr && tries--){
#ifdef OpSysWIN_
            Array = (NumType*)_aligned_malloc(Size * sizeof(NumType), ALIGN);
#elif defined OpSysUNIX_
            Array = (NumType*)aligned_alloc(ALIGN, Size * sizeof(NumType));
#endif
            if (Array == nullptr) Sleep(10);
        }

        AbandonIfNull(Array);
    }

}

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