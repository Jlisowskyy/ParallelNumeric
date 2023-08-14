// Author: Jakub Lisowski
#ifndef __PARALLELNUM_RECURSIVE_FUNCTIONS_H
#define __PARALLELNUM_RECURSIVE_FUNCTIONS_H

#include <functional>
#include <thread>

#include "NumericalCore.hpp"

template<typename NumType, class Operation, NumType InitVal, unsigned (*Decider)(unsigned long long)>
class SmartFuncParent
	// Class template used to define a mathematical sum or product
{
	NumType(*Func)(NumType);
	NumType DistanceHolder = NumType();
	NumType ThreadedResults[MaxCPUThreads] = {0 };
	std::thread* Threads[MaxCPUThreads] = { nullptr };
	Operation Oper;
    void (SmartFuncParent::* Helper)(unsigned long long, NumType, NumType*) = nullptr;


	void HelperFunc(unsigned long long Elements, NumType StartPoint, NumType* Dest);
	void HelperNoFunc(unsigned long long Elements, NumType StartPoint, NumType* Dest);
public:
	SmartFuncParent(NumType(*Func)(NumType) = nullptr);
	NumType operator()(unsigned long long Elements, NumType StartPoint = NumType(), NumType Distance = 1);
};

// -----------------------------
// Smart Func Parent Implementation
// -----------------------------

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(unsigned long long int)>
void
SmartFuncParent<NumType, Operation, InitVal, Decider>::HelperNoFunc(unsigned long long int Elements, NumType StartPoint, NumType *Dest) {
    NumType Res = InitVal;

    while (Elements--) {
        Res = Oper(Res, StartPoint);
        StartPoint += DistanceHolder;
    }

    *Dest = Res;
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(unsigned long long int)>
void SmartFuncParent<NumType, Operation, InitVal, Decider>::HelperFunc(unsigned long long int Elements, NumType StartPoint, NumType *Dest) {
    NumType Res = InitVal;

    while (Elements--) {
        Res = Oper(Res, Func(StartPoint));
        StartPoint += DistanceHolder;
    }

    *Dest = Res;
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(unsigned long long int)>
SmartFuncParent<NumType, Operation, InitVal, Decider>::SmartFuncParent(NumType (*Func)(NumType)) : Func{Func } {
    if (Func == nullptr) {
        Helper = &SmartFuncParent::HelperNoFunc;
    }
    else {
        Helper = &SmartFuncParent::HelperFunc;
    }
}

template<typename NumType, class Operation, NumType InitVal, unsigned int (*Decider)(unsigned long long int)>
NumType SmartFuncParent<NumType, Operation, InitVal, Decider>::operator()(unsigned long long int Elements, NumType StartPoint, NumType Distance) {
    NumType Res = InitVal;
    DistanceHolder = Distance;

    if (Elements <= ThreadedStartingThreshold) {
        (this->*Helper)(Elements, StartPoint, &Res);
        return Res;
    }

    unsigned ThreadCount = Decider(Elements);
    unsigned i;
    unsigned long long ElementsPerThread = Elements / ThreadCount;

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

template <typename T, unsigned ThreadCap = MaxCPUThreads, unsigned (*Decider)(unsigned long long) = LinearThreads<ThreadCap>>
using SmartFuncSum = SmartFuncParent<T, std::plus<T>, 0, Decider>;

template <typename T, unsigned ThreadCap = MaxCPUThreads, unsigned (*Decider)(unsigned long long) = LinearThreads<ThreadCap>>
using SmartFuncProd = SmartFuncParent<T, std::multiplies<T>, 1, Decider>;

#endif