
// Author: Jakub Lisowski

#include "../Include/Management/ResourceManager.hpp"
#include <malloc.h>
#include <exception>

ResourceManager* DefaultMM = nullptr;
size_t MemUsageCollector::GlobalUsage = 0;
ThreadPackage ResourceManager::ThreadAssets[ResourceManager::ThreadsAssetsSetsAmount];

ResourceManager::~ResourceManager(){
    for(auto& Iter : MemAssets){
        delete Iter;
    }
}

ThreadPackage &ResourceManager::GetThreads()
    // TODO: Reconsider this shit
{
    for (auto & ThreadAsset : ThreadAssets) {
        if (!ThreadAsset.Occupied) {
            ThreadAsset.Occupied = true;
            return ThreadAsset;
        }
    }

    throw std::runtime_error("[ERROR] Thread resources overload. All threads should be released after usage\n");
}

void MemUsageCollector::SetUsage(size_t Mem) {
    GlobalUsage -= InstanceUsage;
    GlobalUsage += Mem;
    InstanceUsage = Mem;
}

void MemUsageCollector::AppendUsage(size_t Arg) {
    InstanceUsage += Arg;
    GlobalUsage += Arg;
}

Region::Region(SizeMB Val) : Used{ 0 } {
    Size = Val.GetBytes();
    AllocateCacheAligned();
}

Region::~Region() {
    DeallocateAlignedData();
}

void Region::Reserve(SizeMB Val)
    // Tries to expand region size. Possible only when there are no references to the block.
{
    if (size_t NewSize = Val.GetBytes(); !Used && NewSize > Size) [[likely]]{
        DeallocateAlignedData();
        Size = NewSize;
        AllocateCacheAligned();
    }
}

void Region::AllocateCacheAligned()
    // Corresponding to available system libraries calls proper function to allocate memory aligned to
    // Cache line length. Uses private data member to get the amount of memory to alloc.
{
#ifdef OP_SYS_WIN
    Mem = _aligned_malloc(Size , CacheInfo::LineSize);
#elif defined(OP_SYS_UNIX)
    Mem = aligned_alloc(CacheInfo::LineSize, Size);
#endif

    if (!Mem) [[unlikely]]{
        throw AllocationError(Size, AllocationError::SourceType::RegionAlloc);
    }
}

void Region::DeallocateAlignedData()
    // Corresponding to available system libraries calls proper function to release memory
{
#ifdef OP_SYS_WIN
    _aligned_free(Mem);
#elif defined(OP_SYS_UNIX)
    free(Mem);
#endif
}

