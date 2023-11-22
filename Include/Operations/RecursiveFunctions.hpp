// Author: Jakub Lisowski
#ifndef PARALLEL_NUM_RECURSIVE_FUNCTIONS_H
#define PARALLEL_NUM_RECURSIVE_FUNCTIONS_H

#include <functional>
#include <thread>

#include "NumericalCore.hpp"

// -------------------------------------
// Math sum and Prod core function
// -------------------------------------

template<
        typename NumType,
        class BinOperation,
        NumType InitVal,
        unsigned (*Decider)(ull)>
class SmartFuncParent
	// Class template used to define a mathematical sum or product.
    // Meaning of template parameters:
    // - NumType - type using inside function,
    // - BinOperation - binary operand used inside loop,
    // - InitVal - value used to initialize return value,
    // - Decide - threading strategy functions, returns number of threads that should be used on passed input.
{
// ------------------------------
// Class interaction
// ------------------------------
public:

    SmartFuncParent(NumType(*Func)(NumType) = nullptr);
    NumType operator()(ull Elements, NumType StartPoint = NumType(), NumType Distance = 1);

private:
    void HelperFunc(ull Elements, NumType StartPoint, NumType* Dest);
    void HelperNoFunc(ull Elements, NumType StartPoint, NumType* Dest);

// ------------------------------
// private fields
// ------------------------------

    NumType(*Func)(NumType) { nullptr };
	NumType DistanceHolder { NumType() };
	NumType ThreadedResults[ThreadInfo::MaxCpuThreads] { 0 };
	std::thread* Threads[ThreadInfo::MaxCpuThreads] { nullptr };
	BinOperation Oper;
    void (SmartFuncParent::* Helper)(ull, NumType, NumType*) { nullptr };
};

// -----------------------------
// Smart Func Parent Implementation
// -----------------------------

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(ull)>
SmartFuncParent<NumType, Operation, InitVal, Decider>::SmartFuncParent(NumType (*Func)(NumType)) : Func{Func } {
    if (Func == nullptr) {
        Helper = &SmartFuncParent::HelperNoFunc;
    }
    else {
        Helper = &SmartFuncParent::HelperFunc;
    }
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(ull)>
void SmartFuncParent<NumType, Operation, InitVal, Decider>::HelperNoFunc(ull Elements, NumType StartPoint, NumType *Dest) {
    NumType Res = InitVal;

    while (Elements--) {
        Res = Oper(Res, StartPoint);
        StartPoint += DistanceHolder;
    }

    *Dest = Res;
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(ull)>
void SmartFuncParent<NumType, Operation, InitVal, Decider>::HelperFunc(ull Elements, NumType StartPoint, NumType *Dest) {
    NumType Res = InitVal;

    while (Elements--) {
        Res = Oper(Res, Func(StartPoint));
        StartPoint += DistanceHolder;
    }

    *Dest = Res;
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(ull)>
NumType SmartFuncParent<NumType, Operation, InitVal, Decider>::operator()(ull Elements, NumType StartPoint, NumType Distance) {
    NumType Res = InitVal;
    DistanceHolder = Distance;

    if (Elements <= ThreadInfo::ThreadedStartingThreshold) {
        (this->*Helper)(Elements, StartPoint, &Res);
        return Res;
    }

    unsigned ThreadCount = Decider(Elements);
    unsigned i;
    ull ElementsPerThread = Elements / ThreadCount;

    for (i = 0; i < ThreadCount - 1; ++i) {
        Threads[i] = new std::thread(Helper, *this,
                                     ElementsPerThread, StartPoint, &ThreadedResults[i]);
        StartPoint += ElementsPerThread * Distance;
    }
    Threads[i] = new std::thread(Helper, *this,
                                 Elements - ElementsPerThread * (i), StartPoint, &ThreadedResults[i]);

    for (i = 0; i < ThreadCount; ++i) {
        Threads[i]->join();
        Res = Oper(Res, ThreadedResults[i]);
        delete Threads[i];
    }

    return Res;
}

// -----------------------------
// Basic Interface declaration
// -----------------------------

template <typename T, unsigned ThreadCap = ThreadInfo::MaxCpuThreads, unsigned (*Decider)(ull) = LinearThreads<ThreadCap>>
using SmartFuncSum = SmartFuncParent<T, std::plus<T>, 0, Decider>;

template <typename T, unsigned ThreadCap = ThreadInfo::MaxCpuThreads, unsigned (*Decider)(ull) = LinearThreads<ThreadCap>>
using SmartFuncProd = SmartFuncParent<T, std::multiplies<T>, 1, Decider>;

#endif // PARALLEL_NUM_RECURSIVE_FUNCTIONS_H