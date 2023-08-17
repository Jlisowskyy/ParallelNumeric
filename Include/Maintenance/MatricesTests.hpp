
// Author: Jakub Lisowski

#ifndef PARALLELNUM_MATRICESTESTS_H
#define PARALLELNUM_MATRICESTESTS_H

#include <cstdlib>
#include <cmath>

#include "../Wrappers/OptimalOperations.hpp"

unsigned long long GenerateNumber(unsigned long long MinVal, unsigned long long MaxVal) {
    return  (unsigned long long)((3 * (double)rand() / (double)RAND_MAX) * (double)(MaxVal - MinVal)) + MinVal;
}

struct DPack
    // DimsPack
{
    unsigned dim1, dim2, dim3;
};

DPack GenDims(unsigned long long OpCount){
    unsigned dim1 = GenerateNumber(4, std::cbrt(OpCount));
    unsigned dim2 = GenerateNumber(2, std::sqrt(OpCount / dim1));
    unsigned dim3 = OpCount / (dim1 * dim2);

    return {dim1, dim2, dim3};
}

template <typename NumType, DPack(*GetDims)(unsigned long long) = GenDims>
bool PerformTest(unsigned long long OperationCount, unsigned RunsToDo, long Seed = 0, bool Verbose = false) {

    unsigned SuccessfulRuns = 0;
    long long ShortestRun, LongestRun, LastRun;
    Timer T1("Every Run Counter", false), T2("Only Succesful Runs Counter", false);

    if (!Seed) {
        srand(time(nullptr));
    }

    T1.CalculateAverageTime(RunsToDo, Verbose);
    T2.CalculateAverageTime(RunsToDo);

    for (unsigned i = RunsToDo; i; --i) {

        auto Val1 = (NumType)GenerateNumber(1, 25);
        auto Val2 = (NumType)GenerateNumber(1, 5);
        DPack d = GetDims(OperationCount);

        Matrix1<NumType> M1(d.dim1, d.dim2, Val1);
        Matrix1<NumType> M2(d.dim2, d.dim3, Val2);

        T2.Start();
        T1.Start();
        Matrix1<NumType> M3 = M1 * M2;
        LastRun = T1.Stop();
        T2.Stop();

        if (i == RunsToDo) {
            ShortestRun = LongestRun = LastRun;
        }
        else {
            LongestRun = std::max(LastRun, LongestRun);
            ShortestRun = std::min(LastRun, ShortestRun);
        }

        bool SuccessFlag = M3.CheckForIntegrity((NumType)(d.dim2 * Val1 * Val2), Verbose);

        if (!SuccessFlag) {
            T2.InvalidateLastRun();
        }
        else ++SuccessfulRuns;

        if (Verbose) {
            std::cout << "\nDims:" << d.dim1 << '\n' << d.dim2 << '\n' << d.dim3 << '\n';
        }
    }

    std::cout << "\n\nWith longest time: " << (double)LongestRun * 1e-9 << "(seconds)\nAnd shortest time: "
              << (double)ShortestRun * 1e-9 << "(seconds)\nWith seed: " << Seed << std::endl;

    return SuccessfulRuns == RunsToDo;
}

void PerformMajorTests(unsigned RunsToDo) {
    unsigned CorrectRuns = 0;

    while (RunsToDo--) {
        if (PerformTest<double>(1000000000ull, 100, 0)) {
            ++CorrectRuns;
        }
    }

    std::cout << "\n\n" << CorrectRuns << " of " << RunsToDo << " were successful in major test\n";
}

#endif