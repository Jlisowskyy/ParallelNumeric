
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
class Vector: public MemUsageCollector
{
protected:
    static constexpr size_t ElementsPerCacheLine = CacheInfo::LineSize / sizeof(NumType);
    size_t Size;
	bool IsHorizontal;
	const ResourceManager* MM;
    NumType* Array;

	inline void SetWholeData(NumType Val);
    void CheckForIncorrectSize() const;
	void AllocateArray();
	void DeallocateArray();

	// Used only for init_list<init_list> - unknown parameters
	// Should not be used as a class constructor, may lead to unexpected problems
	Vector(bool IsHorizontal, ResourceManager* MM) noexcept :
        Size{ 0 }, IsHorizontal{ IsHorizontal }, MM{ MM }, Array{ nullptr } {}

public:
#ifdef DEBUG_
    virtual bool CheckForIntegrity(NumType Val, bool Verbose) const;
    virtual bool CheckForIntegrity(NumType* Val, bool Verbose) const;
#endif
	void MoveToArray(std::initializer_list<NumType> Init);

	explicit Vector(size_t Size, bool IsHorizontal = false, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
	}

	Vector(size_t Size, NumType InitVal, bool IsHorizontal = false, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();
		SetWholeData(InitVal);
	}

	Vector(std::initializer_list<NumType> Init, bool IsHorizontal = false, ResourceManager* MM = DefaultMM) noexcept:
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

	Vector(size_t Size, NumType* Init, bool IsHorizontal = false, ResourceManager* MM = DefaultMM) noexcept:
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }, Array{ Init }
	{
        CheckForIncorrectSize();
        BaseAbandonIfNull(Init, 0);
	}

	Vector(size_t Size, const NumType* Init, bool IsHorizontal = false, ResourceManager* MM = DefaultMM) :
		Size{ Size }, IsHorizontal{ IsHorizontal }, MM{ MM }
	{
        CheckForIncorrectSize();
		AllocateArray();

		for (size_t i = 0; i < Size; ++i) {
			Array[i] = Init[i];
		}
	}

    ~Vector(){
        DeallocateArray();
    }

	Vector& operator=(const Vector& Vec);
	Vector& operator=(Vector&& Vec) noexcept;
    inline size_t GetSize() const { return Size; }
	inline bool GetIsHorizontal() const { return IsHorizontal; }
	inline NumType* GetArray() const { return Array; }
    inline NumType* GetArray() { return Array; }
	inline void Transpose() noexcept { IsHorizontal = !IsHorizontal; }

	inline NumType& operator[](size_t Ind) { return Array[Ind]; }
	inline const NumType& operator[](size_t Ind) const { return Array[Ind]; }
private:
	void PrintHorizontally(std::ostream& Out) const;
	void PrintVertically(std::ostream& Out) const;

public:

	friend std::ostream& operator<<(std::ostream& Out, const Vector& Input) {
		if (Input.IsHorizontal)
			Input.PrintHorizontally(Out);
		else
			Input.PrintVertically(Out);

		return Out;
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
        static constexpr size_t PackageSize = AVXInfo::ByteCap / sizeof(NumType);
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
	Vector& sqrt() {
        ApplyOnDataEffect<std::sqrt>();
        return *this;
	}

    Vector& reciprocal(){
        auto Operand = [](NumType x) -> NumType{ return 1 / x; };
        ApplyOnDataEffect<Operand>();
        return *this;
    }

    Vector& rsqrt(){
        auto Operand = [](NumType x) -> NumType{ return 1 / std::sqrt(x); };
        ApplyOnDataEffect<Operand>();
        return *this;
    }

    Vector& exp() {
        ApplyOnDataEffect<std::exp>();
        return *this;
	}

    Vector& exp2() {
        ApplyOnDataEffect<std::exp2>();
        return *this;
	}

    Vector& sin() {
        ApplyOnDataEffect<std::sin>();
        return *this;
	}

    Vector& cos() {
        ApplyOnDataEffect<std::cos>();
        return *this;
	}

    Vector& tan() {
        ApplyOnDataEffect<std::tan>();
        return *this;
	}

    Vector& sinh() {
        ApplyOnDataEffect<std::sinh>();
        return *this;
	}

    Vector& cosh() {
        ApplyOnDataEffect<std::cosh>();
        return *this;
	}

    Vector& tanh() {
        ApplyOnDataEffect<std::tanh>();
        return *this;
	}

    Vector& cot() {
        auto Operand = [](NumType x) -> NumType { return 1 / std::tan(x); };
        ApplyOnDataEffect<Operand>();
        return *this;
    }

    Vector& coth() {
        auto Operand = [](NumType x) -> NumType { return 1 / std::tanh(x); };
        ApplyOnDataEffect<Operand>();
        return *this;
	}

	Vector GetModified(void (Vector::*Func)()) const {
		Vector RetVal = *this;
		(RetVal.*Func)();
		return RetVal;
	}

	Vector GetModified(NumType(*Func)(NumType x)) const {
		Vector RetVal = *this;
        RetVal.ApplyOnDataEffect<Func>();
		return RetVal;
	}


private:
    friend inline DotProductMachine<NumType> GetDProdMach(const Vector<NumType>& A, const Vector<NumType>& B){
        return DotProductMachine<NumType>(A.Array, B.Array, A.Size);
    }
public:
	// Vector and Vector operations

    template<size_t ThreadCap = 20, size_t (*Decider)(size_t) = LinearThreads<ThreadCap>>
    friend NumType operator*(const Vector<NumType>& A, const Vector<NumType>& B){
        if (A.GetIsHorizontal() && !B.GetIsHorizontal()) [[likely]] {

            if (A.GetSize() != B.GetSize()) [[unlikely]] {
                throw std::runtime_error("[ERROR] Vectors are not the same length\n");
            }

            auto Machine = GetDProdMach(A, B);
            return Machine.template Perform<ThreadCap, Decider>();
        }
        else [[unlikely]] {
            throw std::runtime_error("[ERROR] Not matching dimension to perform dot product or they are not the same length\n");
        }
    }

};

template<typename NumType>
void Vector<NumType>::CheckForIncorrectSize() const {
    if (!Size) [[unlikely]]
        throw std::runtime_error("[ERROR] Vector cannot be empty ( 0 size )\n");
}

//-----------------------------------------
// High perf AVX spec
//-----------------------------------------

#ifdef __AVX__

template<>
Vector<double>& Vector<double>::sqrt();
template<>
Vector<float>& Vector<float>::sqrt();
template<>
Vector<float>& Vector<float>::reciprocal();

#endif // __AVX__

//-----------------------------------------
// Template implementation
//-----------------------------------------

#ifdef DEBUG_

template<typename NumType>
bool Vector<NumType>::CheckForIntegrity(NumType *Val, bool Verbose) const {
    for (size_t i = 0; i < Size; ++i)
        if (Array[i] != Val[i]) [[unlikely]]{
            if (Verbose) std::cerr << "[ERROR] Integrity test failed on Index: " << i << '\n';
            return false;
        }

    if (Verbose) std::cout << "Success\n";
    return true;
}

template<typename NumType>
bool Vector<NumType>::CheckForIntegrity(NumType Val, bool Verbose) const {

    for (size_t i = 0; i < Size; ++i)
        if (Array[i] != Val) [[unlikely]] {
            if (Verbose) std::cerr << "[ERROR] Integrity test failed on Index: " << i << '\n';
            return false;
        }

    if (Verbose) std::cout << "Success\n";
    return true;
}

#endif // DEBUG_

template<typename NumType>
void Vector<NumType>::SetWholeData(NumType Val)
    // Sets whole data to desired value in case of Val == 0 uses ZeroMemory macro
{
    if (Val == 0) [[likely]]{
#ifdef OP_SYS_WIN
        ZeroMemory(Array, Size * sizeof(NumType));
#else
        memset(Array, 0, Size * sizeof(NumType));
#endif
    }
    else {
        #pragma omp parallel for
        for (size_t i = 0; i < Size; ++i)
            Array[i] = Val;
    }
}

template<typename NumType>
void Vector<NumType>::DeallocateArray() {
    if (!MM) [[likely]] {
#ifdef OP_SYS_WIN
        _aligned_free(Array);
#elif defined OP_SYS_UNIX
        free(Array);
#endif
    }
    else {
        // TODO
    }
}

template<typename NumType>
void Vector<NumType>::AllocateArray()
    // Allocates memory aligned to cache line length
{
    if (!MM) [[likely]] {
        size_t ByteSize = Size * sizeof(NumType);
#ifdef OP_SYS_WIN
        Array = (NumType*)_aligned_malloc(ByteSize, CacheInfo::LineSize);
#elif defined(OP_SYS_UNIX)
        Array = (NumType*)aligned_alloc(CacheInfo::LineSize, ByteSize);
#endif
        BaseAbandonIfNull(Array, ByteSize);
        SetUsage(ByteSize);
    }
    else {
        // TODO
    }

}

template<typename NumType>
void Vector<NumType>::MoveToArray(std::initializer_list<NumType> Init) {
    if (Init.size() != Size) [[unlikely]]
        throw std::runtime_error("[ERROR] Invalid init list sizes\n");

    const auto Matrix = std::data(Init);

    #pragma omp parallel for
    for (size_t i = 0; i < Size; ++i)
        Array[i] = Matrix[i];
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator=(Vector &&Vec) noexcept {
    auto Temp = Vec.Array;
    Vec.Array = nullptr;
    DeallocateArray();

    Size = Vec.Size;
    IsHorizontal = Vec.IsHorizontal;
    Array = Temp;

    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator=(const Vector &Vec) {
    if (this == &Vec) [[unlikely]] return *this;
    DeallocateArray();

    Size = Vec.Size;
    IsHorizontal = Vec.IsHorizontal;

    AllocateArray();
    memcpy(Array, Vec.Array, Size * sizeof(NumType));

    return *this;
}

template<typename NumType>
void Vector<NumType>::PrintHorizontally(std::ostream &Out) const {
    size_t MaxPerCol = FindConsoleWidth() / 6;

    Out << std::fixed << std::setprecision(3);

    for (size_t i = 0; i + MaxPerCol <= Size; i+=MaxPerCol) {
        Out << "Vector values within index range: " << i << '-' << i + MaxPerCol
            << ':' << std::endl;

        for (size_t j = 0; j < MaxPerCol; ++j) {
            Out << Array[i + j] << ' ';
        }

        Out << std::endl;
    }
}

template<typename NumType>
void Vector<NumType>::PrintVertically(std::ostream &Out) const {
    Out << std::endl;
    for (size_t i = 0; i < Size; ++i)
        Out << Array[i] << '\n';
    Out << std::endl;
}

#endif