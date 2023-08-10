
// Author: Jakub Lisowski
// Prosta biblioteka, realizujaca wielowatkowe obliczenia numeryczna. Zostala stworzona 
// w celach rozrywkowych, przy okazji poznajac mechanizmy wielowatkowosci C++

#ifndef Parallel_Numeric_Jlisowskyy_H
#define Parallel_Numeric_Jlisowskyy_H

#include <thread>
#include <utility>

#define DOUBLE_VECTOR_LENGTH (unsigned long)4
#define FLOAT_VECTOR_LENGTH (unsigned long)8
#define PACKAGE_SIZE (unsigned long)256
#define ALLIGN 64
#define CACHE_LINE 64

#define __DEBUG__

constexpr unsigned long GB = 1024 * 1024 * 1024;

// Detecting system on compilation
#ifdef __unix
    #define __OpSysUNIX__
#elif defined __unix__
    #define __OpSysUNIX__
#elif defined __linux__
    #define __OpSysUNIX__
#elif  defined _WIN32
    #define __OpSysWIN__
#elif defined _WIN64
    #define __OpSysWIN__
#else
    #define __OpSysNONE__
#endif

// Correct including behaviour
#ifdef __OpSysUNIX__

#include <sys/ioctl.h> 
    
int FindConsoleWidth() {
    winsize buff;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &buff);

    return buff.ws_col;
}

#elif defined __OpSysWIN__

#include <windows.h>
#include <sysinfoapi.h>


#else
    // Solve problem
#endif
    //TODO: SOLVE PROBLEMS WITH DETECTIONS


// Temporary
#define MaxCPUThreads (unsigned)18
#define TotalHWMemory (unsigned long)32768
#define MaxMemUsage (unsigned long)24576

int FindConsoleWidth(); 

#endif