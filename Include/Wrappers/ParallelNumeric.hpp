
// Author: Jakub Lisowski
// Prosta biblioteka, realizujaca wielowatkowe obliczenia numeryczna. Zostala stworzona 
// w celach rozrywkowych i edukacyjnych, przy okazji poznajac mechanizmy wielowatkowosci C++

#ifndef PARALLEL_NUMERIC_JLISOWSKYY_H
#define PARALLEL_NUMERIC_JLISOWSKYY_H

#include <thread>
#include <utility>

// ----------------------------------
// Compilation controllers
// ----------------------------------

#define DEBUG_

// TODO: Not done yet not all operations adapt to this optimisation
// #define OPTIMISE_MEM_

// Globally adjust some functions to by the default emit double specialized types
using DefaultNumType = double;

// ----------------------------------

#ifdef __AVX__

namespace AVXInfo
{
    constexpr size_t f64Cap { 4 };
    constexpr size_t f32Cap { 8 };
    constexpr size_t BitCap { 256 };
    constexpr size_t ByteCap { 256 / 8 };

    template<typename NumType>
    constexpr size_t GetAVXLength(){
        return ByteCap / sizeof(NumType);
    }
}

#endif

namespace CacheInfo
    // Sizes defined in Byte manner
{
    constexpr size_t LineSize { 64 };
    constexpr size_t L1Size { 32 * 1024 };
    constexpr size_t L2Size { 2 * 1024 * 1024 }; // Per core assumption
    constexpr size_t L3Size { 20 * 1024 * 1024 }; // Global assumption
}

using cun = const unsigned;
using cull = const unsigned long long;
using ull = unsigned long long;

namespace ThreadInfo{
    constexpr size_t ThreadedStartingThreshold { 32768 };
    constexpr size_t BasicThreadPool { 8 };
    constexpr size_t MaxCpuThreads { 20 };
}

namespace MemoryInfo{
    constexpr size_t GB { 1024 * 1024 * 1024 } ;
    constexpr size_t MB { 1024 * 1024 };
    constexpr size_t KB { 1024 };
    constexpr size_t TotalHwMem { 32768 };
    constexpr size_t MaxMemUsage { 24576 };
}


// Detecting the system on compilation
// TODO: Move to CMake
#ifdef __unix
    #define OP_SYS_UNIX
#elif defined __unix__
    #define OP_SYS_UNIX
#elif defined __linux__
    #define OP_SYS_UNIX
#elif  defined _WIN32
    #define OP_SYS_WIN
#elif defined _WIN64
    #define OP_SYS_WIN
#else
    #define OpSysNONE_
#endif

// Correct including behavior
#ifdef OP_SYS_UNIX

#include <sys/ioctl.h> 

#elif defined(OP_SYS_WIN)

#include <windows.h>
#include <sysinfoapi.h>

#else
    // Solve problem
#endif
    //TODO: SOLVE PROBLEMS WITH DETECTIONS


int FindConsoleWidth();

//template<typename NumType>
//inline constexpr size_t GetAccCount()
//    // Used in function initialization, assumes 8 is neutral for compiler loop unrolling in different algorithm
//{
//    return 8;
//}

template<typename NumType>
inline constexpr size_t GetCacheLineElem()
{
    return CacheInfo::LineSize / sizeof(NumType);
}

#ifdef DEBUG_
#include "../Maintenance/Debuggers.hpp"
#endif

#endif // PARALLEL_NUMERIC_JLISOWSKYY_H