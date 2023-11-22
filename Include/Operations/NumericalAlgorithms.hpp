// Author: Jakub Lisowski

#ifndef PARALLELNUM_ALGORITHMS_H
#define PARALLELNUM_ALGORITHMS_H
#include "../Wrappers/OptimalOperations.hpp"

/*          Notes:
 * There will be some implementations of complex or not numerical algorithms implemented mainly,
 * using solutions from the projects.
 * Including GE, LU decompositions, integrals, approximation and interpolation.
 */

// TODO: Repair after changes

//template<typename T>
//T AdaptiveRomberg1(T(*Func)(T), T StartPoint, T StopPoint,
//	unsigned short NumOfNodes = 15, ResourceManager& MM = DefaultManager) {
//	SmartFuncSum<T> SumMachine(Func);
//	T SectionLength = (StopPoint - StartPoint), Sum;
//	T* RombergTab;
//
//	MM.ArrayPop(&RombergTab, NumOfNodes);
//	if (Func != nullptr)
//		RombergTab[0] = (Func(StartPoint) + Func(StopPoint)) * SectionLength;
//	else // Temporary
//		RombergTab[0] = (StartPoint + StopPoint) * SectionLength;
//
//	ull Pow = 1;
//	for (size_t i = 1; i < NumOfNodes; ++i) {
//		Sum = SumMachine(Pow, StartPoint + SectionLength / 2, SectionLength) * SectionLength;
//
//		RombergTab[i] = (T)0.5 * (RombergTab[i - 1] + Sum);
//		SectionLength /= 2;
//		Pow *= 2;
//	}
//
//	Pow = 4;
//	for (size_t i = 1; i < NumOfNodes; ++i) {
//		for (unsigned short j = NumOfNodes - 1; j >= i; --j) {
//			RombergTab[j] = (Pow * RombergTab[j] - RombergTab[j - 1]) / (Pow - 1);
//		}
//		Pow *= 4;
//	}
//
//	return RombergTab[NumOfNodes - 1];
//}

#endif // PARALLELNUM_ALGORITHMS_H