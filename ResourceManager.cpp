
// Author: Jakub Lisowski

#include "ResourceManager.hpp"
#include <malloc.h>
#include <exception>

unsigned long ResourceManager::AllUsedMemory = 0;
unsigned short ResourceManager::ExistingInstances = 0;
ThreadPackage ResourceManager::ThreadAssets[ResourceManager::ThreadsAssetsSetsAmount];


ResourceManager::ResourceManager(){
    ++ExistingInstances;
    if (!GetRegion(OperatingRegionSize))
#ifdef _MSC_VER 
        throw std::exception("Not able to init OpMemory");
#else
        throw std::exception();
#endif
}

ResourceManager::~ResourceManager(){
    for (short i = 0; i < MemoryAssetsSize; ++i)
        if (MemoryAssets[i] != nullptr) delete MemoryAssets[i];

    AllUsedMemory -= UsedMemory;
    --ExistingInstances;
}

void ResourceManager::SortMemoryAssets() {
    int sorted = 0;
    for (int i = 0; i < MemoryAssetsSize; ++i) {
        if (MemoryAssets[i] != nullptr) {
            MemoryAssets[sorted++] = MemoryAssets[i];
            MemoryAssets[i] = nullptr;
        }
    }

    MemoryAssetsInd = sorted;
}

Region* ResourceManager::GetRegion(unsigned SizeMB)
// Possibly to return nullptr when new not succeed
{
    if (SizeMB + AllUsedMemory > MaxMemUsage || UsedRegions >= MemoryAssetsSize) {
        return nullptr;
    }

    if (!(MemoryAssetsInd < MemoryAssetsSize)) SortMemoryAssets();

    Region* RetPtr = new (std::nothrow) Region(SizeMB);

    if (RetPtr != nullptr) {
        UsedMemory += SizeMB;
        AllUsedMemory += SizeMB;
        ++UsedRegions;
        MemoryAssets[MemoryAssetsInd++] = RetPtr;
    }

    return RetPtr;
}

void ResourceManager::DeleteRegion(Region* RegionToRemove)
{
    if (RegionToRemove == nullptr) return;

    for (int i = 0; i < MemoryAssetsSize; ++i)
        if (RegionToRemove == MemoryAssets[i]) {
            UsedMemory -= (unsigned long)(MemoryAssets[i]->Size / Region::MB);
            AllUsedMemory -= (unsigned long)(MemoryAssets[i]->Size / Region::MB);
            --UsedRegions;

            delete MemoryAssets[i];
            MemoryAssets[i] = nullptr;
            return;
        }

   
#ifdef _MSC_VER 
    throw std::exception("Region managment fault RSMG-DeleteRegion");
#else
    throw std::exception();
#endif
}