cmake_minimum_required(VERSION 3.26)
project(ParallelNumeric)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(ParallelNumeric
        main.cpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/ErrorCodes.hpp
        Include/Maintenance/MatricesTests.hpp
        Include/Management/ResourceManager.hpp
        Include/Operations/MatrixAndVectorOperations.hpp
        Include/Operations/NumericalAlgorithms.hpp
        Include/Operations/NumericalCore.hpp
        Include/Operations/RecursiveFunctions.hpp
        Include/Types/Matrix.hpp
        Include/Types/Vector.hpp
        Include/Wrappers/InterfaceFunctions.hpp
        Include/Wrappers/ParallelNumeric.hpp
        Include/Wrappers/OptimalOperations.hpp
        Include/Operations/MatrixMultiplicationSolutions.hpp
)

# dodaj zeby podczas kompilacji nie includowalo

target_compile_options(ParallelNumeric PUBLIC -Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-mavx;-mavx2;-mfma;-O3)
