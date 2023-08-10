
// Author: Jakub Lisowski

#include <cmath>
#include <windows.h>

#include "OptimalOperations.hpp"
#include "Debuggers.hpp"
#include "MatricesTests.hpp"

//#define __DebugSumProd
#define __DebugMatrix1
//#define __DebugVectors1

#ifdef __DebugSumProd

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
#elif defined __DebugVectors1


using typior = double;

int main() {
    unsigned size = 16384;
    Vector<typior> V1(size, (typior)4, false);
    Vector<typior> V2(size, (typior)1);
    Timer T1;

    Matrix1<typior> result = OuterProduct(V1, V2);

    //std::cout << result;
}

#elif defined __DebugMatrix1


using typ = double;
unsigned dim1 = 1000;
unsigned dim2 = 1000;
unsigned dim3 = 1000;
typ Val = 6;


int main() {
    Matrix1<typ> M1(dim1, dim2, (typ)1);
    Matrix1<typ> M2(dim2, dim3, Val);

    Timer T1(nullptr, false);
    
    auto M3 = M1 * M2;
    T1.Stop();

    M3.CheckForIntegrity((typ)(Val * dim2), true);

    //PerformTest<double>(1000000000ull, 100, 7, true);

    ////PerformMajorTests(3);
}


// Before changes exec time was: 2.94 - 3333/1111
// Before changes exec time was: 0.6 - 1000/1000
// Actual time is: 

#endif