
// Author: Jakub Lisowski

#ifndef PARALLELNUMERIC_RESOURCEMANAGER_HPP
#define PARALLELNUMERIC_RESOURCEMANAGER_HPP

#include <malloc.h>
#include <exception>
#include <cmath>
#include <list>

#include "../Wrappers/ParallelNumeric.hpp"
#include "../Maintenance/ErrorCodes.hpp"

template<size_t ThreadCap = ThreadInfo::MaxCpuThreads>
inline size_t LogarithmicThreads(size_t);
    // Simple Example of Thread number calculation, used in all threaded functions

template<size_t ThreadCap = ThreadInfo::MaxCpuThreads>
inline size_t LinearThreads(size_t);
    // Simple Example of Thread number calculation, used in all threaded functions

class MemUsageCollector
    // Used in memory usage optimizations and debugging information also
    // Collects information about memory usage in specific case of allocating memory without ResourceManager
{
    static size_t GlobalUsage;
protected:
    size_t InstanceUsage;
    void SetUsage(size_t Mem);
    void AppendUsage(size_t Arg);
public:
    MemUsageCollector(): InstanceUsage { 0 } {}
    ~MemUsageCollector() { GlobalUsage -= InstanceUsage; }
    static size_t GetGlobalUsage() { return GlobalUsage; }
};

class SizeMB
    // Interface reminding that memory should be passed in MegaBytes
{
    size_t Size;
public:
    explicit SizeMB(size_t Val): Size{ Val } {}
    size_t GetBytes() const { return Size * MemoryInfo::MB; }
};

class Region{
    size_t Size;
    size_t Used;
    void* Mem;

    class SubRegion{
        size_t Size;
        size_t Used;
        void* Mem;

        std::list<SubRegion> SubRegions;
    public:
        void Expand(const size_t Val) { Size += Val; }
    };

    std::list<SubRegion> SubRegions;

    inline void AllocateCacheAligned();
    inline void DeallocateAlignedData();
    friend class ResourceManager;
    public:
        explicit Region(SizeMB Val);
        ~Region();

        void Reserve(SizeMB Val);

        template<typename PrimitiveT>
        PrimitiveT* Allocate(size_t AllocSize);

        template<typename PrimitiveT>
        PrimitiveT* AllocateAligned(size_t AllocSize, size_t Alignment);
};


struct ThreadPackage {
    std::thread* Array[ThreadInfo::MaxCpuThreads]{ nullptr };
    bool Occupied = false;

    ThreadPackage() = default;

    inline void Release() { Occupied = false; }
};

class ResourceManager: public MemUsageCollector
    // Class used to globally manage memory usage, provides less system call allocation to improve performance, but
    // may also lead to increased memory usage
{
    static constexpr size_t ThreadsAssetsSetsAmount { 16 };
    static ThreadPackage ThreadAssets[ThreadsAssetsSetsAmount];
public:
    static ThreadPackage& GetThreads();

private:
    static constexpr size_t MemAssetsSize { 64 };
    size_t UsedMemory { 0 };
    size_t UsedRegions { 0 };
    size_t MemoryAssetsInd { 0 };

    Region* MemAssets[MemAssetsSize] { nullptr };

public:
    ~ResourceManager();
};

template<size_t ThreadCap>
size_t LogarithmicThreads(const size_t Elements) {
    auto Ret = static_cast<size_t>(log2(Elements / ThreadInfo::ThreadedStartingThreshold + 1));
    return std::min(ThreadCap, Ret);
}

template<size_t ThreadCap>
size_t LinearThreads(const size_t Elements) {
    auto Ret = static_cast<size_t>(Elements / ThreadInfo::ThreadedStartingThreshold);
    return std::min(ThreadCap, Ret);
}

template<typename PrimitiveT>
PrimitiveT* Region::Allocate(size_t AllocSize)
    // Returns unaligned chunk of memory stored inside the region
    // If AllocSize exceeds region capacity returns nullptr
{
    if (size_t NewUsage = Used + AllocSize * sizeof(PrimitiveT); NewUsage <= Size) {
        PrimitiveT* RetVal = (size_t)Mem + Used;
        Used = NewUsage;
        return RetVal;
    }
    else return nullptr;
}

template<typename PrimitiveT>
PrimitiveT *Region::AllocateAligned(size_t AllocSize, size_t Alignment)
    // Returns aligned chunk of memory stored inside the region
{
    const size_t OverPop = Used % Alignment;
    const size_t AlignmentOffset = Alignment - OverPop;
    if (const size_t NewUsage = AllocSize * sizeof(PrimitiveT) + AlignmentOffset + Used; NewUsage <= Size){
        const size_t RetOffset = Used + Alignment - OverPop;
        Used = NewUsage;
        return (size_t)Mem + RetOffset;
    }
    else return nullptr;
}

#endif