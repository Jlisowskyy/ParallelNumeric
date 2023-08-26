
// Author: Jakub Lisowski

#ifndef PARALLELNUM_MATRICESTESTS_H
#define PARALLELNUM_MATRICESTESTS_H

#include <cstdlib>
#include <cmath>

#include "../Wrappers/OptimalOperations.hpp"

size_t GenerateNumber(size_t MinVal, size_t MaxVal) {
    return  (size_t)((3 * (double)rand() / (double)RAND_MAX) * (double)(MaxVal - MinVal)) + MinVal;
}

struct DPack
    // DimsPack
{
    size_t dim1, dim2, dim3;
};

DPack GenDims(size_t OpCount){
    size_t dim1 = GenerateNumber(4, std::cbrt(OpCount));
    size_t dim2 = GenerateNumber(2, std::sqrt(OpCount / dim1));
    size_t dim3 = OpCount / (dim1 * dim2);

    return {dim1, dim2, dim3};
}

template <typename NumType = double , DPack(*GetDims)(size_t) = GenDims>
bool PerformMMTest(size_t OperationCount, unsigned RunsToDo, long Seed = 0, bool Verbose = false) {

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

    bool SuccessFlag = SuccessfulRuns == RunsToDo;
    if (Verbose) {
        if (SuccessFlag) {
            std::cout << "All runs were successful\n";
        } else {
            std::cout << "[ERROR] PROBLEM OCCURRED NOT ALL RUNS WERE SUCCESSFUL\n";
        }
    }
    return SuccessFlag;
}

template<typename NumType, void(Vector<NumType>::*UnaryOperand)()>
void PerformVectOnDataTest(size_t VectorSize, size_t RunsToDo){
    Timer T1("Every Run Counter", false);
    T1.CalculateAverageTime(RunsToDo, true);
    Vector<NumType> V1(VectorSize, (NumType)100);

    while(RunsToDo--){
        T1.Start();
        (V1.*UnaryOperand)();
        T1.Stop();
    }
}

void PerformMajorTests(unsigned RunsToDo) {
    unsigned CorrectRuns = 0;

    while (RunsToDo--) {
        if (PerformMMTest<double>(1000000000ull, 100, 0)) {
            ++CorrectRuns;
        }
    }

    std::cout << "\n\n" << CorrectRuns << " of " << RunsToDo << " were successful in major test\n";
}

template <typename NumType = double , DPack(*GetDims)(size_t) = GenDims>
bool PerformOPTest(size_t OperationCount, unsigned RunsToDo, long Seed = 0, bool IsHor = false, bool Verbose = false){
    unsigned SuccessfulRuns = 0;
    long long ShortestRun, LongestRun, LastRun;
    Timer T1("Every Run Counter", false);

    if (!Seed) {
        srand(time(nullptr));
    }

    T1.CalculateAverageTime(RunsToDo, Verbose);

    for (unsigned i = RunsToDo; i; --i) {
        auto Val1 = (NumType)GenerateNumber(1, 25);
        auto Val2 = (NumType)GenerateNumber(1, 5);
        DPack d = GetDims(OperationCount);

        Vector<NumType> V1(d.dim1, Val1, false);
        Vector<NumType> V2(d.dim2, Val2, true);

        T1.Start();
        auto M = GetOuterProduct(V1,V2);
        LastRun = T1.Stop();

        if (i == RunsToDo) {
            ShortestRun = LongestRun = LastRun;
        }
        else {
            LongestRun = std::max(LastRun, LongestRun);
            ShortestRun = std::min(LastRun, ShortestRun);
        }

        bool SuccessFlag = M.CheckForIntegrity(Val1 * Val2, Verbose);
        if (SuccessFlag) ++SuccessfulRuns;
    }

    std::cout << "\n\nWith longest time: " << (double)LongestRun * 1e-9 << "(seconds)\nAnd shortest time: "
              << (double)ShortestRun * 1e-9 << "(seconds)\nWith seed: " << Seed << std::endl;

    bool SuccessFlag = SuccessfulRuns == RunsToDo;
    if (Verbose) {
        if (SuccessFlag) {
            std::cout << "All runs were successful\n";
        } else {
            std::cout << "[ERROR] PROBLEM OCCURRED NOT ALL RUNS WERE SUCCESSFUL\n";
        }
    }
    return SuccessFlag;
}

#endif