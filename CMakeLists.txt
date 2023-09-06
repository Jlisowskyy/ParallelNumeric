cmake_minimum_required(VERSION 3.26)
project(ParallelNumeric)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(gcc_flag "$<COMPILE_LANG_AND_ID:CXX,GNU,LCC>")
set(clang_flag "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang>")

#set(-DCMAKE_TOOLCHAIN_FILE C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake)
#find_package(superlu CONFIG REQUIRED)
#target_link_libraries(main PRIVATE superlu::superlu)
#find_package(arpackng CONFIG REQUIRED)
#target_link_libraries(main PRIVATE ARPACK::ARPACK)
#find_package(Armadillo CONFIG REQUIRED)
#target_link_libraries(main PRIVATE armadillo)


add_executable(ParallelNumeric
        main.cpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/Debuggers.hpp
        Include/Maintenance/ErrorCodes.hpp
        Include/Maintenance/PerfTests.hpp
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
        Src/ParallelNumeric.cpp
        Src/ResourceManager.cpp
        Src/Debuggers.cpp
        Src/Matrix.cpp
        Src/NumericalCore.cpp
        Src/ErrorCodes.cpp
        Src/MatrixMultiplicationSolutions.cpp
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

set(NEEDED_FLAGS "-mavx;-mavx2;-mfma;-masm=intel")
set(WARNING_FLAGS "-Wall;-Wextra;-Wformat=2;-Wunused;")
set(OPTIMIZATION_FLAGS "-fopenmp;-march=native;-O3")
set(DEBUG_FLAGS "-g;-Rpass-analysis=inline")

target_compile_options(ParallelNumeric PUBLIC ${NEEDED_FLAGS} ${WARNING_FLAGS} ${OPTIMIZATION_FLAGS})
