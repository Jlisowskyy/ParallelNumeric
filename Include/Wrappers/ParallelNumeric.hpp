// Author: Jakub Lisowski
// Simple library performing multi-threaded numerical operations.
// Was created in educational and entertainment purposes only.
// The goal was to explore some architectural designs and multi-threaded c++ libraries.

#ifndef PARALLEL_NUMERIC_JLISOWSKYY_H
#define PARALLEL_NUMERIC_JLISOWSKYY_H

#include <thread>
#include <utility>

// ----------------------------------
// Compilation controllers
// ----------------------------------

#define DEBUG_

/*              IMPORTANT NOTES ON ACTUAL PROJECT STATE:
 *  - currently it is not possible to correctly perform all operations with different memory layout,
 *    because not all of those procedures are already implemented.
 *  - OPTIMISE_MEM_ should not be used yet, because it also needs re-implementing some of the operation functions.
 *  - AVX optimisations are not implemented yet for all supported types.
 *  - Not everything is properly tested yet, for example, some default not supported types operations.
 *  - Global memory and thread managing structure should be implemented.
 *  - Hardware detecting procedures should be introduced.
 *  - MEMORY MANAGEMENT SHOULD NOT BE USED YET/
 * */

// WARNING: Not working correctly yet.
// Note: there are some cache alignment optimizations done, which introduces enormous memory usage overhead for some
//       edge cases, for example, the size of 2 * 1e+9 matrices will be increased by 4 times.
//       This optimization should fix this problem in the future, but will probably introduce some small global overhead.

// #define OPTIMISE_MEM_

// Globally adjust some functions to by the default emit double specialized types
using DefaultNumType = double;

// ---------------------------------------
// Hardware related constants values and functions
// ---------------------------------------

#ifdef __AVX__

namespace AVXInfo
    // Some AVX related constants
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

#endif // __AVX__

namespace CacheInfo
    // Sizes defined in Byte manner
    // Note: Currently cache size is hard-coded.
    // TODO: allow hardware detecting.
{
    constexpr size_t LineSize { 64 };
    constexpr size_t L1Size { 32 * 1024 };
    constexpr size_t L2Size { 2 * 1024 * 1024 }; // Per core assumption
    constexpr size_t L3Size { 20 * 1024 * 1024 }; // Global assumption
}

namespace ThreadInfo
    // Note: Currently threading information is hard-coded.
    // TODO: allow hardware detecting.
{
    constexpr size_t ThreadedStartingThreshold { 32768 };
    constexpr size_t BasicThreadPool { 8 };
    constexpr size_t MaxCpuThreads { 20 };
}

namespace MemoryInfo
    // Note: Currently memory information is hard-coded.
    // TODO: allow hardware detecting.
{
    constexpr size_t GB { 1024 * 1024 * 1024 } ;
    constexpr size_t MB { 1024 * 1024 };
    constexpr size_t KB { 1024 };
    constexpr size_t TotalHwMem { 32768 };
    constexpr size_t MaxMemUsage { 24576 };
}

template<typename NumType>
inline constexpr size_t GetCacheLineElem()
{
    return CacheInfo::LineSize / sizeof(NumType);
}

// ------------------------------
// Globally used aliases
// ------------------------------

using cun = const unsigned;
using cull = const unsigned long long;
using ull = unsigned long long;

// ----------------------------------------
// Os detecting preprocessor commands
// ----------------------------------------

// TODO: Move to CMake?
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

// ---------------------------------------------
// Environment based conditional including
// ---------------------------------------------

// Correct OS-based including behavior
//TODO: SOLVE PROBLEMS WITH DETECTIONS
#ifdef OP_SYS_UNIX
    #include <sys/ioctl.h>
#elif defined(OP_SYS_WIN)
    #include <windows.h>
    #include <sysinfoapi.h>
#else
    #error Not possible to correctly detect OS
#endif

#ifdef DEBUG_
    #include "../Maintenance/Debuggers.hpp"
#endif

// -------------------------------------
// Hardware/OS depending functions
// -------------------------------------

int FindConsoleWidth();

#endif // PARALLEL_NUMERIC_JLISOWSKYY_H