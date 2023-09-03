
// Author: Jakub Lisowski

#ifndef PARALLELNUM_ERRORS_H_
#define PARALLELNUM_ERRORS_H_

#include <cstdlib>
#include <iostream>
#include <string>
#include "../Wrappers/ParallelNumeric.hpp"

void BaseAbandonIfNull(void *Ptr, size_t ToAlloc);

class AllocationError: public std::runtime_error{
public:
    enum class SourceType { RegionAlloc, OnRunAlloc };

private:
    SourceType Type;
    size_t ToAlloc;

    static std::string WhatArg;
public:
    AllocationError(size_t ToAlloc, SourceType T);
    ~AllocationError(){ WhatArg = "[ERROR] During allocation occurred specific problem:\n"; }
};

#endif