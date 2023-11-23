cmake_minimum_required(VERSION 3.26)
project(ParallelNumeric)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(ParallelNumeric
        main.cpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/ErrorCodes.hpp
        Include/Maintenance/PerfTests.hpp
        Include/Management/ResourceManager.hpp
        Include/Operations/NumericalAlgorithms.hpp
        Include/Operations/NumericalCore.hpp
        Include/Operations/RecursiveFunctions.hpp
        Include/Types/Matrix.hpp
        Include/Types/Vector.hpp
        Include/Wrappers/ParallelNumeric.hpp
        Include/Wrappers/OptimalOperations.hpp
        Src/ParallelNumeric.cpp
        Src/ResourceManager.cpp
        Src/Debuggers.cpp
        Src/Matrix.cpp
        Src/NumericalCore.cpp
        Src/ErrorCodes.cpp
        Include/Operations/MatrixMultiplication.hpp
        Src/MatrixMultiplication.cpp
        Src/Vector.cpp
        Include/Operations/AVXSolutions.hpp
        Include/Management/ThreadSolutions.hpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelNumeric PUBLIC OpenMP::OpenMP_CXX)
endif()

set(NEEDED_FLAGS "-mavx;-mavx2;-mfma;-masm=intel;")
set(WARNING_FLAGS "-Wall;-Wextra;-Wformat=2;-Wunused;")
set(OPTIMIZATION_FLAGS "-fopenmp;-march=native;-O3")
set(DEBUG_FLAGS "-g;-Rpass-analysis=inline;")

target_compile_options(ParallelNumeric PUBLIC ${NEEDED_FLAGS} ${WARNING_FLAGS} ${OPTIMIZATION_FLAGS})
