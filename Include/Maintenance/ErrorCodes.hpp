
// Author: Jakub Lisowski

#ifndef PARALLEL_NUM_ERRORS_H_
#define PARALLEL_NUM_ERRORS_H_

#include <cstdlib>
#include <iostream>
#include <string>
#include "../Wrappers/ParallelNumeric.hpp"

// ------------------------------
// Error functions
// ------------------------------

void BaseAbandonIfNull(void *Ptr, size_t ToAlloc);

// ------------------------------
// Error classes
// ------------------------------

class AllocationError final: public std::runtime_error{
    // ------------------------------
    // Class Interaction
    // ------------------------------
public:
    enum class SourceType { RegionAlloc, OnRunAlloc };

    AllocationError(size_t ToAlloc, SourceType T);
    ~AllocationError() final { WhatArg = "[ERROR] During allocation occurred specific problem:\n"; }

    // ------------------------------
    // Private fields
    // ------------------------------
private:
    SourceType Type;
    size_t ToAlloc;
    static std::string WhatArg;
};

#endif // PARALLEL_NUM_ERRORS_H_