// Author: Jakub Lisowski

#ifndef PARALLELNUM_MATRICES_TESTS_H
#define PARALLELNUM_MATRICES_TESTS_H

#include <cstdlib>
#include <cmath>
#include <tuple>
#include <functional>

#include "../Wrappers/OptimalOperations.hpp"

// ------------------------------------------
// Basic templates and tests components
// ------------------------------------------

size_t GenerateNumber(size_t MinVal, size_t MaxVal)
    // Generates random number from range [MinVal, MaxVal], used mainly in tests to generate random input sizes
{
    return (size_t)((3 * (double)rand() / (double)RAND_MAX) * (double)(MaxVal - MinVal)) + MinVal;
}

// Used to pass dimensions of inputs inside tests
using D3Pack = std::tuple<size_t, size_t, size_t>;
using D2Pack = std::tuple<size_t, size_t>;

D3Pack Gen3Dims(size_t OpCount)
    // Generates random 3 elements vector, with given property: As used as dimensions in matrix multiplication
    // basic matrix algorithm should have close to OpCount operation complexity (dim1 * dim2 * dim3 ~~ OpCount)
{
    size_t dim1 = GenerateNumber(4, (size_t)std::cbrt(OpCount));
    size_t dim2 = GenerateNumber(2, (size_t)std::sqrt(OpCount / dim1));
    size_t dim3 = OpCount / (dim1 * dim2);

    return std::make_tuple(dim1, dim2, dim3);
}

D2Pack Gen2Dims(size_t OpCount)
    // Generates random 2 elements vector, with given property:
    // As used as dimensions in matrix and vector multiplication
    // basic algorithm should have close to OpCount operation complexity (dim1 * dim2 ~~ OpCount)
{
    size_t dim1 = GenerateNumber(2, (size_t)std::sqrt(OpCount));
    size_t dim2 = GenerateNumber(2, OpCount / dim1);

    return std::make_tuple(dim1, dim2);
}

template<
        typename Arg1T,
        typename Arg2T,
        typename ResT,
        std::tuple<Arg1T, Arg2T> (*ConstrPack)(size_t),
        ResT (*FuncToTest)(std::tuple<Arg1T, Arg2T>&),
        bool (*SuccessTest)(std::tuple<Arg1T, Arg2T>&, ResT&, bool) = nullptr,
        bool Verbose = false
        >
void PerformXXXTest(size_t OpCount, unsigned RunsToGo, long Seed = 0)
    // Template functions used to give highly customizable foundation to different performance and quality tests
    // of desired binary or unary operation.
    // On the end prints average operation time and if is verbose prints also
    // amount of failed and successful attempts.
    // Meaning of template arguments:
    // - Arg1T - type of first argument,
    // - Arg2T - type of second argument,
    // - ResT - type of result,
    // - ConstrPack - function which creates and returns tuple of constructed and properly prepared first and second
    //                arguments used later in tests, should use argument as an indication of arguments sizes.
    // - FuncToTest - function responsible for applying operation on arguments tuple, returns operation result
    // - SuccessTest - function used on both arguments and result to test them.
    // Should return a test result.
    //                 On the last argument there will be passed Verbose bool flag, indicating whether function should
    //                 write something on console or not.
    // - Verbose - Flag indicating whether function should write something, during the test, on console or not.
    // Meaning of function arguments:
    // - OpCount - argument passed into ConstrPack function
    // - RunsToGo - how many times operations should be applied, each other time ConstrPack is invoked again.
    // - Seed - by default 0, this means time function is used in srand, otherwise passed Seed is used.
{
    size_t SuccessfulRuns{};
    long long ShortestRun{};
    long long LongestRun{};
    long long LastRun{};
    Timer T1("Every Run Counter", false), T2("Only Successful Runs Counter", false);

    srand(Seed == 0 ? time(nullptr) : Seed);

    T1.CalculateAverageTime(RunsToGo, Verbose);
    T2.CalculateAverageTime(RunsToGo);

    for (size_t i = RunsToGo; i; --i) {
        std::tuple<Arg1T, Arg2T> Args = ConstrPack(OpCount);

        T2.Start();
        T1.Start();
        auto Result = FuncToTest(Args);
        LastRun = T1.Stop();
        T2.Stop();

        if (i == RunsToGo) {
            ShortestRun = LongestRun = LastRun;
        }
        else {
            LongestRun = std::max(LastRun, LongestRun);
            ShortestRun = std::min(LastRun, ShortestRun);
        }

        if constexpr(SuccessTest){
            bool SuccessFlag = SuccessTest(Args, Result, Verbose);
            if (!SuccessFlag) {
                T2.InvalidateLastRun();
            }
            else {
                ++SuccessfulRuns;
            }
        }
    }

    std::cout << "\n\nWith longest time: " << (double)LongestRun * 1e-9 << "(seconds)\nAnd shortest time: "
              << (double)ShortestRun * 1e-9 << "(seconds)\nWith seed: " << Seed << std::endl;

    if constexpr (Verbose) {
        if (SuccessfulRuns == RunsToGo) {
            std::cout << "All runs were successful\n";
        } else {
            std::cout << "[ERROR] PROBLEM OCCURRED NOT ALL RUNS WERE SUCCESSFUL\n";
        }

        std::cout << "Successful runs: " << SuccessfulRuns << "\nFailed runs: " << RunsToGo - SuccessfulRuns << std::endl;
    }
}

// ---------------------------------
// Matrix multiplication tests
// ---------------------------------

template<
        bool Verbose = false,
        D3Pack(*GetDims)(size_t) = Gen3Dims,
        bool IsArg1Hor = false,
        bool IsArg2Hor = false,
        bool IsResultHor = false,
        typename NumType = DefaultNumType
        >
void PerformMMTest(size_t OpCount, unsigned RunsToGo, long Seed = 0)
    // Performs matrix multiplication tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - GetDims - function responsible for generating matrices dimensions used in operation.
    // OpCount argument will be
    //             passed to it.
    // - IsArg1Hor - should first argument be stored row-wise,
    // - IsArg2Ver - should second argument be stored row-wise,
    // - IsResultHor - should the result be stored row-wise,
    // - NumType - type used in Matrix class template.
{
    using MatT = Matrix<NumType>;
    using MatP = std::tuple<MatT, MatT>;

    PerformXXXTest<MatT, MatT, MatT,
                    [](size_t OpCount) -> MatP{
                        NumType Val1 = GenerateNumber(1,25);
                        NumType Val2 = GenerateNumber(1, 5);
                        D3Pack d = GetDims(OpCount);

                        if constexpr (Verbose){
                            std::cout << "Generated Dims: " << std::get<0>(d) << ", " << std::get<1>(d) <<
                                    ", " << std::get<2>(d) << '\n';
                        }

                        return std::make_tuple(
                                MatT(std::get<0>(d), std::get<1>(d), Val1, IsArg1Hor),
                                MatT(std::get<1>(d), std::get<2>(d), Val2, IsArg2Hor)
                        );
                    },
                    [](MatP& Args) -> MatT{
                        return operator*(std::get<0>(Args), std::get<1>(Args));
                    },
                    [](MatP& Args, MatT& Result, bool Verb) -> bool{
                        NumType ExpectedRes = std::get<0>(Args).GetCols() * std::get<0>(Args)[0] * std::get<1>(Args)[0];
                        return Result.CheckForIntegrity(ExpectedRes, Verb);
                    },
                    Verbose>(OpCount, RunsToGo, Seed);
}

// ------------------------------------
// Vector on data operation tests
// ------------------------------------

template<
        typename NumType,
        Vector<NumType>&(Vector<NumType>::*UnaryOperand)(),
        bool Verbose = false
        >
void PerformVectOnDataTest(size_t VectorSize, unsigned RunsToGo)
    // Performs simple performance test of passed UnaryOperand function accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // NumType - type used inside Vector class,
    // Vector<NumType>::*UnaryOperand - method of class vector which will be tested.
{
    using VecT = Vector<NumType>;
    using VecP = std::tuple<VecT, bool>;

    PerformXXXTest<VecT, bool, VecT&,
            [](size_t VSize) -> VecP{
                return std::make_tuple(VecT(VSize, (NumType)100), true);
            },
            [](VecP& Args) -> VecT&{
                return (std::get<0>(Args).*UnaryOperand)();
            },
            nullptr, Verbose
        >(VectorSize, RunsToGo);
}

// ------------------------------
// Outer product tests
// ------------------------------

template<
        bool Verbose = false,
        bool IsHor = false,
        D2Pack(*GetDims)(size_t) = Gen2Dims,
        typename NumType = DefaultNumType
        >
void PerformOutProdTest(size_t OpCount, unsigned RunsToGo, long Seed = 0)
    // Performs outer product operation tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - GetDims - function responsible for generating result matrix dimensions.
    // - IsHor - should output matrix be horizontally stored or not.
    // - NumType - type used in Matrix and Vector class template.
{
    using ArgT = Vector<NumType>;
    using ArgP = std::tuple<ArgT, ArgT>;
    using RetT = Matrix<NumType>;

    PerformXXXTest<ArgT, ArgT, RetT,
            [](size_t OpCount) -> ArgP{
                NumType Val1 = GenerateNumber(1,25);
                NumType Val2 = GenerateNumber(1, 5);

                D2Pack Dims = GetDims(OpCount);

                if constexpr(Verbose){
                    std::cout << "Generated dims: " << std::get<0>(Dims) << ", " << std::get<1>(Dims) << '\n';
                }

                return std::make_tuple(ArgT(std::get<0>(Dims), Val1, false),
                                       ArgT(std::get<1>(Dims), Val2, true));
            },
            [](ArgP& Args) -> RetT{
                return GetOuterProduct(std::get<0>(Args), std::get<1>(Args), IsHor);
            },
            [](ArgP& Args, RetT& RetVal, bool Verb) -> bool{
                return RetVal.CheckForIntegrity(std::get<0>(Args)[0] * std::get<1>(Args)[0], Verb);
            },Verbose>(OpCount, RunsToGo, Seed);
}

// ------------------------------------------
// Vector & Matrix multiplication tests
// ------------------------------------------

template<
        bool Verbose = false,
        bool IsArgMatHor = false,
        bool IsVectHor = false,
        D2Pack(*GetDims)(size_t) = Gen2Dims,
        typename NumType = DefaultNumType
        >
void PerformVectMatMultTest(size_t OperationCount, unsigned RunsToGo, long Seed = 0)
    // Performs matrix & vector multiplication tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - GetDims - function responsible for generating matrices dimensions used in operation.
    //             OpCount argument will be passed to it.
    // - IsArgMatHor - should argument matrix be stored horizontally.
    // - IsVectHor - determines which type of operation should be done: vect * matrix or matrix * vect.
    // - NumType - type used in Matrix and Vector class template.
{
    using MatT = Matrix<NumType>;
    using VecT= Vector<NumType>;
    using ArgP = std::tuple<MatT, VecT>;

    PerformXXXTest<MatT, VecT, VecT,
            [](size_t OpCount) -> ArgP{
                size_t VectDim{};
                D2Pack Dims = GetDims(OpCount);
                NumType Val1 = GenerateNumber(2, 25);
                NumType Val2 = GenerateNumber(1,5);

                if constexpr (Verbose){
                    std::cout << "Generated dimensions: " << std::get<0>(Dims) << ", " << std::get<1>(Dims) << '\n';
                }

                // When vector is transposed another dimension should be equal
                if constexpr (IsVectHor) VectDim = std::get<0>(Dims);
                else VectDim = std::get<1>(Dims);

                return std::make_tuple(MatT(std::get<0>(Dims), std::get<1>(Dims), Val1, IsArgMatHor),
                                       VecT(VectDim, Val2, IsVectHor));
            },
            [](ArgP& Args) -> VecT{
                if constexpr(IsVectHor)
                    // Vector * Matrix
                {
                    return std::get<1>(Args) * std::get<0>(Args);
                }
                else
                    // Matrix * Vector
                {
                    return std::get<0>(Args) * std::get<1>(Args);
                }
            },
            [](ArgP& Args, VecT& Result, bool Verb) -> bool{
                size_t CoDim{};
                if constexpr(IsVectHor){
                    CoDim = std::get<0>(Args).GetRows();
                }
                else{
                    CoDim = std::get<0>(Args).GetCols();
                }

                return Result.CheckForIntegrity(CoDim * std::get<0>(Args)[0] * std::get<1>(Args)[0], Verb);
            }, Verbose>
            (OperationCount, RunsToGo, Seed);
}

// ------------------------------
// Inner product tests
// ------------------------------

template<
        bool Verbose = false,
        typename NumType = DefaultNumType
        >
void PerformInnerProductTest(size_t VectorSize, unsigned RunsToGo)
    // Performs inner product operation tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - NumType - type used in Matrix and Vector class template.
    // Additional function arguments:
    // - VectorSize - determines the complexity of performed operations.
{
    using VecT = Vector<NumType>;
    using VecP = std::tuple<VecT, VecT>;

    PerformXXXTest<VecT, VecT, NumType,
            [](size_t VSize) -> VecP{
                return std::make_tuple(VecT(VSize, 1, true), VecT(VSize, 2, false));
            },
            [](VecP& Args) -> NumType{
                return std::get<0>(Args) * std::get<1>(Args);
            },
            [](VecP& Args, NumType& Result, bool){
                auto ExpectedValue { static_cast<NumType>(std::get<0>(Args).GetSize() * 2) };
                bool RetCond { Result == ExpectedValue };

                if constexpr(Verbose){
                    std::cout << "Acquired result: " << Result << "\nExpected Result: " << ExpectedValue << '\n';

                    if (RetCond) std::cout << "Success!!\n";
                    else std::cout << "This run was not successful one!!\n";
                }

                return RetCond;
            }, Verbose
    >(VectorSize, RunsToGo);
}

// ------------------------------
// Matrix sum tests
// ------------------------------

template<
        bool Verbose = false,
        D2Pack(*GetDims)(size_t) = Gen2Dims,
        bool IsResHor = false,
        bool IsArg1Hor = false,
        bool IsArg2Hor = false,
        typename NumType = DefaultNumType
        >
void PerformMatrixSumTest(size_t OpCount, unsigned RunsToGo)
    // Performs matrix sum tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - GetDims - function responsible for generating matrices dimensions used in operation.
    //             OpCount argument will be passed to it.
    // - IsArg1Hor - should first argument be stored row-wise,
    // - IsArg2Ver - should second argument be stored row-wise,
    // - IsResHor - should the result be stored row-wise,
    // - NumType - type used in Matrix class template.
{
    using MatT = Matrix<NumType>;
    using ArgP = std::tuple<MatT, MatT>;

    PerformXXXTest<MatT, MatT, MatT,
            [](size_t OpCount) -> ArgP{
                D2Pack Dims { GetDims(OpCount) };

                if constexpr (Verbose){
                    std::cout << "Generated dims: " << std::get<0>(Dims) << ", " << std::get<1>(Dims) << '\n';
                }

                NumType Val1 { GenerateNumber(1,500) };
                NumType Val2 { GenerateNumber(1, 500) };

                return std::make_tuple(MatT(std::get<0>(Dims), std::get<1>(Dims), Val1, IsArg1Hor),
                                       MatT(std::get<0>(Dims), std::get<1>(Dims), Val2, IsArg2Hor));
            },
            [](ArgP& Args) -> MatT{
                return std::get<0>(Args) + std::get<1>(Args);
            },
            [](ArgP& Args, MatT& Result, bool Verb){
                NumType ExpectedValue { std::get<0>(Args)[0] + std::get<1>(Args)[0] };
                return Result.CheckForIntegrity(ExpectedValue, Verb);
            }, Verbose
    >(OpCount, RunsToGo);
}

// -----------------------------------
// Crossed array operation tests
// -----------------------------------

template<
        typename NumType,
        NumType (*BinOperand)(NumType, NumType),
        bool Verbose = false,
        D2Pack(*GetDims)(size_t) = Gen2Dims,
        bool IsResHor = false,
        bool IsArg1Hor = false,
        bool IsArg2Hor = false
        >
void performCrossedArrayTest(size_t OpCount, unsigned RunsToGo)
    // Performs the element by element matrix-to-matrix operation tests accordingly to PerformXXXTest function.
    // For details, refer to their description.
    // Additional template parameters:
    // - BinOperand - operation applied to 'crossed arrays' (Matrices with different memory layout).
    // - GetDims - function responsible for generating matrices dimensions used in operation.
    //             OpCount argument will be passed to it.
    // - IsArg1Hor - should first argument be stored row-wise,
    // - IsArg2Ver - should second argument be stored row-wise,
    // - IsResHor - should the result be stored row-wise,
    // - NumType - type used in Matrix class template.
{
    using MatT = Matrix<NumType>;
    using ArgP = std::tuple<MatT, MatT>;

    PerformXXXTest<MatT, MatT, MatT,
            [](size_t OpCount) -> ArgP{
                D2Pack Dims { GetDims(OpCount) };

                if constexpr (Verbose){
                    std::cout << "Generated dims: " << std::get<0>(Dims) << ", " << std::get<1>(Dims) << '\n';
                }

                NumType Val1 { GenerateNumber(1,500) };
                NumType Val2 { GenerateNumber(1, 500) };

                return std::make_tuple(MatT(std::get<0>(Dims), std::get<1>(Dims), Val1, IsArg1Hor),
                                       MatT(std::get<0>(Dims), std::get<1>(Dims), Val2, IsArg2Hor));
            },
            [](ArgP& Args) -> MatT{
// -------------------- TODO: INSERT TEMPLATE -------------------------------------------------------------------------------------------------------
                return std::get<0>(Args) + std::get<1>(Args);
            },
            [](ArgP& Args, MatT& Result, bool Verb){
                NumType ExpectedValue { BinOperand(std::get<0>(Args)[0], std::get<1>(Args)[0]) };
                return Result.CheckForIntegrity(ExpectedValue, Verb);
            }, Verbose
    >(OpCount, RunsToGo);
}

#endif // PARALLELNUM_MATRICES_TESTS_H