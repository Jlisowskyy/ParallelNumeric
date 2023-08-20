
// Author: Jakub Lisowski

#include <cmath>
#include <windows.h>

#include "/Include/Maintenance/Debuggers.hpp"
#include "/Include/Maintenance/MatricesTests.hpp"
#include "/Include/Wrappers/OptimalOperations.hpp"

//#define DebugSumProd
#define DebugMatrix1
//#define DebugVectors1

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

int main() {
    unsigned size = 16384;
    Vector<typior> V1(size, (typior)4, false);
    Vector<typior> V2(size, (typior)1);
    Timer T1;

    Matrix1<typior> result = OuterProduct(V1, V2);

    //std::cout << result;
}

#elif defined DebugMatrix1

const unsigned dim = 4080;

int main() {
    PerformTest<double, [](unsigned long long opCount) -> DPack{ return {dim,dim,dim} ;}>(1e+9, 1, 69);
}

// BEFORE BLOCKING UPDATE: 5.5 on dim dim dim, where dim = 4096
// AFTER BLOCKING UPDATE: ___

#endif