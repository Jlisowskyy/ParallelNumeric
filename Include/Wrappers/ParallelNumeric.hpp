
// Author: Jakub Lisowski
// Prosta biblioteka, realizujaca wielowatkowe obliczenia numeryczna. Zostala stworzona 
// w celach rozrywkowych, przy okazji poznajac mechanizmy wielowatkowosci C++

#ifndef Parallel_Numeric_Jlisowskyy_H
#define Parallel_Numeric_Jlisowskyy_H

#include <thread>
#include <utility>

#define DOUBLE_VECTOR_LENGTH 4
#define SINGLE_VECTOR_LENGTH 8
#define BYTE_SIZE 8
#define AVX_SIZE 256
#define ALIGN 64
#define CACHE_LINE 64

// Compilation controllers
#define DEBUG_
//#define OPTIMISE_MEM_

using cun = const unsigned;
using cull = const unsigned long long;
using ull = unsigned long long;

const long long unsigned ThreadedStartingThreshold = 32768;
const unsigned BasicThreadPool = 8;
const unsigned MaxCPUThreads = 20;
constexpr unsigned long GB = 1024 * 1024 * 1024;

// Detecting the system on compilation
#ifdef __unix
    #define OpSysUNIX_
#elif defined __unix__
    #define OpSysUNIX_
#elif defined __linux__
    #define OpSysUNIX_
#elif  defined _WIN32
    #define OpSysWIN_
#elif defined _WIN64
    #define OpSysWIN_
#else
    #define OpSysNONE_
#endif

// Correct including behavior
#ifdef OpSysUNIX_

#include <sys/ioctl.h> 

#elif defined OpSysWIN_

#include <windows.h>
#include <sysinfoapi.h>


#else
    // Solve problem
#endif
    //TODO: SOLVE PROBLEMS WITH DETECTIONS


// Temporary
#define TOTAL_HW_MEMORY (unsigned long)32768
#define MAX_MEM_USAGE (unsigned long)24576
#define MATRIX_MULT_BLOCK_COEF 4

int FindConsoleWidth(); 

#endif