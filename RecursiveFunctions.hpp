// Author: Jakub Lisowski
#ifndef __PARALLELNUM_RECURSIVE_FUNCTIONS_H
#define __PARALLELNUM_RECURSIVE_FUNCTIONS_H

#include <functional>
#include <thread>

#include "NumericalCore.hpp"

template<typename T, class Operation, T InitVal, class Y = LinearThreads>
class SmartFuncParent
	// Class template used to define mathematical sum or product
{
	T(*Func)(T);
	T DistanceHolder = T();
	T ThreadedResults[MaxCPUThreads] = { 0 };
	std::thread* Threads[MaxCPUThreads] = { nullptr };
	Y ThreadDecider;
	Operation Oper;

	void HelperFunc(unsigned long long Elements, T StartPoint, T* Dest) {
		T Res = InitVal;

		while (Elements--) {
			Res = Oper(Res, Func(StartPoint));
			StartPoint += DistanceHolder;
		}

		*Dest = Res;
	}

	// Make a specialized template for Sum function and replace it with normal algo XD
	void HelperNoFunc(unsigned long long Elements, T StartPoint, T* Dest) {
		T Res = InitVal;

		while (Elements--) {
			Res = Oper(Res, StartPoint);
			StartPoint += DistanceHolder;
		}

		*Dest = Res;
	}

	void (SmartFuncParent::* Helper)(unsigned long long, T, T*) = nullptr;

public:
	SmartFuncParent(T(*Func)(T) = nullptr) : Func{ Func } {
		if (Func == nullptr) {
			Helper = &SmartFuncParent::HelperNoFunc;
		}
		else {
			Helper = &SmartFuncParent::HelperFunc;
		}
	}

	T operator()(unsigned long long Elements, T StartPoint = T(), T Distance = 1) {
		T Res = InitVal;
		DistanceHolder = Distance;

		if (Elements <= ThreadDecider.StartingThreshold) {
			(this->*Helper)(Elements, StartPoint, &Res);
			return Res;
		}

		unsigned ThreadCount = ThreadDecider(Elements), i;
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
};

template <typename T, class Y = LinearThreads>
using SmartFuncSum = SmartFuncParent<T, std::plus<T>, 0, Y>;

template<typename T, class Y = LinearThreads>
using SmartFuncProd = SmartFuncParent<T, std::multiplies<T>, 1, Y>;

#endif