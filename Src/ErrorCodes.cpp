//
// Created by Jlisowskyy on 14/08/2023.
//
#include "../Include/Maintenance/ErrorCodes.hpp"
#include "../Include/Management/ResourceManager.hpp"

std::string AllocationError::WhatArg = "[ERROR] During allocation occurred specific problem:\n";

// ----------------------------------
// Error classes implementation
// ----------------------------------

AllocationError::AllocationError(size_t ToAlloc, AllocationError::SourceType T) :
        std::runtime_error(
                [&]() -> std::string&{
                    WhatArg = WhatArg + "Allocation type: " + (T == SourceType::OnRunAlloc ? "on run allocation\n" : "region allocation\n")
                            + "Wanted to allocate: " + std::to_string(ToAlloc)
                            + "\nWhen memory allocated was: " + std::to_string(MemUsageCollector::GetGlobalUsage())
                            + "\nDesired summed allocation size: " + std::to_string(ToAlloc + MemUsageCollector::GetGlobalUsage())
                            + "\nWhen available memory is: " + std::to_string(MemoryInfo::TotalHwMem * MemoryInfo::MB) + '\n';

                    return WhatArg;
                }()
        ),
        Type{ T }, ToAlloc { ToAlloc }
{}

// ------------------------------------
// Error functions implementation
// ------------------------------------

void BaseAbandonIfNull(void *Ptr, size_t ToAlloc) {
    if (Ptr == nullptr) {
        throw AllocationError(ToAlloc, AllocationError::SourceType::OnRunAlloc);
    }
}