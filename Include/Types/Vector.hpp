
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
	Vector(
        bool IsHorizontal,
        ResourceManager* MM
    ) noexcept :
        Size{ 0 },
        IsHorizontal{ IsHorizontal },
        MM{ MM },
        Array{ nullptr } {}

public:
#ifdef DEBUG_
    virtual bool CheckForIntegrity(NumType Val, bool Verbose) const;
    virtual bool CheckForIntegrity(NumType* Val, bool Verbose) const;
#endif
	void MoveToArray(std::initializer_list<NumType> Init);

	explicit Vector(
            size_t Size,
            bool IsHorizontal = false,
            ResourceManager* MM = DefaultMM
    ) noexcept;

	Vector(
        size_t Size,
        NumType InitVal,
        bool IsHorizontal = false,
        ResourceManager* MM = DefaultMM
    ) noexcept;

	Vector(
        std::initializer_list<NumType> Init,
        bool IsHorizontal = false,
        ResourceManager* MM = DefaultMM
    ) noexcept;

	Vector(
        const Vector& Target
    ) noexcept;

	Vector(
        Vector&& Target
    ) noexcept;

	Vector(
        size_t Size,
        NumType* Init,
        bool IsHorizontal = false,
        ResourceManager* MM = DefaultMM
    ) noexcept;

	Vector(
        size_t Size,
        const NumType* Init,
        bool IsHorizontal = false,
        ResourceManager* MM = DefaultMM
    );

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
    inline void ApplyOnDataEffect();

#if defined(__AVX__) || defined(__AVX2__)
    template<
            typename AVXType,
            AVXType (AVXOperation)(AVXType),
            NumType(*UnCleaningOperation)(NumType)
            >
    inline void ApplyAVXOnDataEffect();
#endif // __AVX__ or __AVX2__

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

    template<
            size_t ThreadCap = 20,
            size_t (*Decider)(size_t) = LinearThreads<ThreadCap>
            >
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

// --------------------------------------
// TODO: reconsider this part:
// --------------------------------------

    // TODO: make universal template for +/-

private:
    template<NumType (*BinOp)(NumType, NumType)>
    friend inline Vector<NumType> GetArrOnArrResult(const Vector<NumType>& A, const Vector<NumType>& B){
        Vector<NumType> RetVal(A.Size);
        return ApplyArrayOnArrayOp<NumType, BinOp>(RetVal.Array, A.Array, B.Array, B.Size);
    }

    template<NumType (*BinOp)(NumType, NumType)>
    friend inline Vector<NumType> GetScalOnArrResult(const Vector<NumType>& A, NumType& B){
        Vector<NumType> RetVal(A.Size);
        return ApplyScalarOpOnArray<NumType, BinOp>(RetVal.Array, A.Array, B, A.Size);
    }
public:

    Vector<NumType>& operator+=(const Vector<NumType>& B);
    Vector<NumType>& operator+=(const NumType& B);
    Vector<NumType>& operator-=(const Vector<NumType>& B);
    Vector<NumType>& operator-=(const NumType& B);
    Vector<NumType>& operator*=(const NumType& B);
    Vector<NumType>& operator/=(const NumType& B);
    Vector<NumType>& ApplyElemByElemMult(const Vector<NumType>& B);
    Vector<NumType>& ApplyElemByElemDiv(const Vector<NumType>& B);

    friend Vector<NumType> operator+(const Vector<NumType>& A, const Vector<NumType>& B){
        if (A.Size != B.Size){
            throw std::runtime_error("[ERROR] Not able to perform Vector addition: Not same sizes\n");
        }

        return GetArrOnArrResult<std::plus<NumType>>(A, B);
    }

    friend Vector<NumType> operator+(const Vector<NumType>& A, const NumType& B){
        if (A.Size != B.Size){
            throw std::runtime_error("[ERROR] Not able to perform Vector addition: Not same sizes\n");
        }

        return GetScalOnArrResult<std::plus<NumType>>(A, B);
    }

    friend Vector<NumType> operator-(const Vector<NumType>& A, const Vector<NumType>& B){
        if (A.Size != B.Size){
            throw std::runtime_error("[ERROR] Not able to perform Vector substitution: Not same sizes\n");
        }
        return GetArrOnArrResult<std::minus<NumType>>(A, B);
    }

    friend Vector<NumType> operator-(const Vector<NumType>& A, const NumType& B){
        if (A.Size != B.Size){
            throw std::runtime_error("[ERROR] Not able to perform Vector substitution: Not same sizes\n");
        }
        return GetScalOnArrResult<std::minus<NumType>>(A, B);
    }

    friend Vector<NumType> operator*(const Vector<NumType>& A, const NumType& B){
        return GetScalOnArrResult<std::multiplies<NumType>>(A, B);
    }

    friend Vector<NumType> operator/(const Vector<NumType>& A, const NumType& B){
        if (B == 0){ // Todo: reconsider sort of switch to disable this check or else
            throw std::runtime_error("[ERROR] Division by zero\n");
        }

        const NumType Coef { 1 / B };
        return GetScalOnArrResult<std::multiplies<NumType>>(A, Coef);
    }

    friend Vector<NumType> ElemByElemMult(const Vector<NumType>& A, const Vector<NumType>& B){
        if (A.Size != B.Size) [[unlikely]]{
            throw std::runtime_error("[ERROR] Not able to perform ElemByElemMult, because Vectors have different sizes\n");
        }
        return GetArrOnArrResult<std::multiplies<NumType>>(A,B);
    }

    friend Vector<NumType> ElemByElemDiv(const Vector<NumType>& A, const Vector<NumType>& B){
        if (A.Size != B.Size) [[unlikely]]{
            throw std::runtime_error("[ERROR] Not able to perform ElemByElemDiv, because Vectors have different sizes\n");
        }
        return GetArrOnArrResult<std::divides<NumType>>(A,B);
    }
};

//-----------------------------------------
// Construction helpers implementation
//-----------------------------------------

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
        // To unify allocation and algorithms,
        // all vectors like matrices are extended to sizes divisible by length of cache line
        const size_t ElementsOnLastCacheLine = Size % GetCacheLineElem<NumType>();
        const size_t ElementsToExtend { ElementsOnLastCacheLine == 0 ? 0 : (GetCacheLineElem<NumType>() - ElementsOnLastCacheLine) };
        const size_t ExtendedSize { Size + ElementsToExtend };
        const size_t ByteSize { ExtendedSize * sizeof(NumType) } ;
#ifdef OP_SYS_WIN
        Array = (NumType*)_aligned_malloc(ByteSize, CacheInfo::LineSize);
#elif defined(OP_SYS_UNIX)
        Array = (NumType*)aligned_alloc(CacheInfo::LineSize, ByteSize);
#endif
        BaseAbandonIfNull(Array, ByteSize);
        SetUsage(ByteSize);

        // TODO: Temporary solution
        for (size_t j = Size; j < ExtendedSize; ++j) Array[j] = 0;
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

//-----------------------------------------------------------
// Constructors implementation
//-----------------------------------------------------------

template<typename NumType>
Vector<NumType>::Vector(size_t Size, const NumType *Init, bool IsHorizontal, ResourceManager *MM):
        Size{ Size },
        IsHorizontal{ IsHorizontal },
        MM{ MM }
{
    CheckForIncorrectSize();
    AllocateArray();

    for (size_t i = 0; i < Size; ++i) {
        Array[i] = Init[i];
    }
}

template<typename NumType>
Vector<NumType>::Vector(size_t Size, NumType *Init, bool IsHorizontal, ResourceManager *MM) noexcept:
        Size{ Size },
        IsHorizontal{ IsHorizontal },
        MM{ MM },
        Array{ Init }
{
    CheckForIncorrectSize();
    BaseAbandonIfNull(Init, 0);
}

template<typename NumType>
Vector<NumType>::Vector(Vector &&Target) noexcept:
        Size{ Target.Size },
        IsHorizontal{ Target.IsHorizontal },
        MM{ Target.MM },
        Array{ Target.Array }
{
    Target.Array = nullptr;
}

template<typename NumType>
Vector<NumType>::Vector(const Vector &Target) noexcept:
        Size{ Target.Size },
        IsHorizontal{ Target.IsHorizontal },
        MM{ Target.MM }
{
    AllocateArray();
    memcpy(Array, Target.Array, Target.Size * sizeof(NumType));
}

template<typename NumType>
Vector<NumType>::Vector(std::initializer_list<NumType> Init, bool IsHorizontal, ResourceManager *MM) noexcept:
        Size{ Init.size() },
        IsHorizontal{ IsHorizontal },
        MM{ MM }
{
    CheckForIncorrectSize();
    AllocateArray();
    MoveToArray(Init);
}

template<typename NumType>
Vector<NumType>::Vector(size_t Size, NumType InitVal, bool IsHorizontal, ResourceManager *MM) noexcept:
        Size{ Size },
        IsHorizontal{ IsHorizontal }, MM{ MM }
{
    CheckForIncorrectSize();
    AllocateArray();
    SetWholeData(InitVal);
}

template<typename NumType>
Vector<NumType>::Vector(size_t Size, bool IsHorizontal, ResourceManager *MM) noexcept:
        Size{ Size },
        IsHorizontal{ IsHorizontal },
        MM{ MM }
{
    CheckForIncorrectSize();
    AllocateArray();
}

template<typename NumType>
void Vector<NumType>::CheckForIncorrectSize() const {
    if (!Size) [[unlikely]]
        throw std::runtime_error("[ERROR] Vector cannot be empty ( 0 size )\n");
}

//----------------------------------------
// Vector apply operators implementation
//----------------------------------------

template<typename NumType>
template<NumType (*UnaryOperation)(NumType)>
void Vector<NumType>::ApplyOnDataEffect()
// Transforms data using templated function
{
    #pragma omp parallel for
    for (size_t i = 0; i < Size; ++i) {
        Array[i] = UnaryOperation(Array[i]);
    }
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::ApplyElemByElemDiv(const Vector<NumType> &B) {
    if (Size != B.Size) [[unlikely]]{
        throw std::runtime_error("[ERROR] Not able to perform ApplyElemByElemDiv, because Vectors have different sizes\n");
    }
    ApplyArrayOnArrayOp<NumType, std::multiplies<NumType>>(Array, Array, B.Array, B.Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::ApplyElemByElemMult(const Vector<NumType> &B) {
    if (Size != B.Size) [[unlikely]]{
        throw std::runtime_error("[ERROR] Not able to perform ApplyElemByElemMult, because Vectors have different sizes\n");
    }
    ApplyArrayOnArrayOp<NumType, std::multiplies<NumType>>(Array, Array, B.Array, B.Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator/=(const NumType &B) {
    if (B == 0){ // Todo: reconsider sort of switch to disable this check or else
        throw std::runtime_error("[ERROR] Division by zero\n");
    }

    const NumType Coef { 1 / B };
    ApplyScalarOpOnArray<NumType, std::multiplies<NumType>>(Array, Array, Coef, Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator*=(const NumType &B) {
    ApplyScalarOpOnArray<NumType, std::multiplies<NumType>>(Array, Array, B, Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator-=(const NumType &B) {
    return *this += (-B);
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator-=(const Vector<NumType> &B) {
    if (Size != B.Size) [[unlikely]]{
        throw std::runtime_error("[ERROR] Not able to perform Vector substitution: not same size\n");
    }
    ApplyArrayOnArrayOp<NumType, std::minus<NumType>>(Array, Array, B.Array, B.Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator+=(const NumType &B) {
    ApplyScalarOpOnArray<NumType, std::plus<NumType>>(Array, Array, B, Size);
    return *this;
}

template<typename NumType>
Vector<NumType> &Vector<NumType>::operator+=(const Vector<NumType> &B) {
    if (Size != B.Size) [[unlikely]]{
        throw std::runtime_error("[ERROR] Not able to perform Vector addition: not same size\n");
    }
    ApplyArrayOnArrayOp<NumType, std::plus<NumType>>(Array, Array, B.Array, B.Size);
    return *this;
}

//-----------------------------------------
// AVX specs of applying operators and functions
//-----------------------------------------

#if defined(__AVX__) || defined(__AVX2__)

template<typename NumType>
template<
        typename AVXType,
        AVXType (*AVXOperation)(AVXType),
        NumType (*UnaryCleaningOperation)(NumType)
        >
void Vector<NumType>::ApplyAVXOnDataEffect()
// Transforms data with avx function
{
    static constexpr size_t PackageSize = AVXInfo::ByteCap / sizeof(NumType);
    const size_t Range = (Size / GetCacheLineElem<NumType>()) * GetCacheLineElem<NumType>();

#pragma omp parallel for
    for (size_t i = 0; i < Range; i+= GetCacheLineElem<NumType>()) {
        *((AVXType*)(Array + i)) = AVXOperation(*((AVXType*)(Array + i)));
        *((AVXType*)(Array + i + PackageSize)) = AVXOperation(*((AVXType*)(Array + i + PackageSize)));
    }
    for(size_t i = Range; i < Size; ++i){
        Array[i] = UnaryCleaningOperation(Array[i]);
    }
}


#endif // AVX OR AVX2

#ifdef __AVX__

template<>
Vector<double>& Vector<double>::sqrt();
template<>
Vector<float>& Vector<float>::sqrt();
template<>
Vector<float>& Vector<float>::reciprocal();

#endif // __AVX__

//-----------------------------------------
// Debug helpers and printing
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

//-----------------------------------------
// Assignment operators implementation
//-----------------------------------------

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

#endif