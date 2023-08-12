
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
#define ALIGN 64
#define CACHE_LINE 64

#define DEBUG_

constexpr unsigned long GB = 1024 * 1024 * 1024;

// Detecting system on compilation
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

// Correct including behaviour
#ifdef OpSysUNIX_

#include <sys/ioctl.h> 
    
int FindConsoleWidth() {
    winsize buff;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &buff);

    return buff.ws_col;
}

#elif defined OpSysWIN_

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