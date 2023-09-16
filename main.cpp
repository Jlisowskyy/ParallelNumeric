
// Author: Jakub Lisowski

#include <cmath>

#ifdef OpSysWin_
#include <windows.h>
#endif

#include "Include/Maintenance/Debuggers.hpp"
#include "Include/Maintenance/PerfTests.hpp"
#include "Include/Wrappers/OptimalOperations.hpp"

//#define DebugSumProd
//#define DebugMatrix1
#define DebugVectors1

#ifdef DebugSumProd

double MyFunc(double x) {
    return exp(-x) * sqrt(x) + 1;
}

int main(){
    double result;
    SmartFuncProd<double> Product;
    auto time_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i)
        result = AdaptiveRomberg1<double>(sin, 0, 2, 30);

    auto time_stop = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed <<"Spent time: " << (time_stop.time_since_epoch() - time_start.time_since_epoch()).count()/(1000000 * 10)
        << "\n Result: " << result;
}
#elif defined DebugVectors1

using typior = double;
constexpr size_t length = (512 + 256) * 1024 * 1024;
constexpr size_t size = 8160;
constexpr size_t SlowingSize = 8192;

int main() {
    PerformMMTest<true, [](size_t)->D3Pack{ return std::make_tuple(1024, 1024, 1024); }>(length, 2, 4);
}

#elif defined DebugMatrix1

  const unsigned dim = 8160;

int main() {
//    Mat M1(dim, dim, (double)0);
//    Mat M2(dim, dim, (double)1);

//    Timer T1;
//    auto M3 = M1 * M2;
//    T1.Stop();
//    std::cout << M3;

    PerformMMTest<double, [](unsigned long long) -> D3Pack { return {dim,dim,dim}; } >(1e+9, 5, 15060);
}

// BEFORE BLOCKING UPDATE: 5.5 on dim dim dim, where dim = 4096
// AFTER BLOCKING UPDATE: ___

#endif