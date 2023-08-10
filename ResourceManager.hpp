
// Author: Jakub Lisowski

#ifndef PARALLELNUM_RESOURCE_MANAGER_H
#define PARALLELNUM_RESOURCE_MANAGER_H

#include "ParallelNumeric.hpp"
#include <malloc.h>
#include <exception>

class Region{
    static constexpr int MB = 1024 * 1024;
    long long unsigned Size;
    long long unsigned Used;
    void* Memory;

    friend class ResourceManager;
    public:
        Region(unsigned SizeMB) : Used{ 0 } {
            Size = MB * SizeMB;
            Memory = malloc(Size);

            if (Memory == nullptr){
#ifdef _MSC_VER 
                throw std::exception("Allocation error");
#else
                throw std::exception();
#endif           
            }
        }

        ~Region(){
            free(Memory);
        }

        template<typename T>
        void Allocate(T** AllocRes, long long unsigned AllocSize){
            if (Used + AllocSize * sizeof(T) > Size)
                *AllocRes = nullptr;
            else{
                *AllocRes = (T*)((char*)Memory + Used);
                Used += AllocSize * sizeof(T);
            }
        }        

};

//class Linker {
//    Linker** Manager = nullptr;
//    bool Alive;
//    bool Occupied;
//public:
//    Linker(bool Alive = false, bool Occupied = false) : Alive{ Alive }, Occupied{ Occupied } {}
//    bool IsAvailable() {
//        return Alive == false || Occupied == false;
//    }
//    friend class LinkManager;
//};
//
//class LinkManager {
//    Linker* Connections[2];
//public:
//    LinkManager(): Connections {nullptr} {}
//    ~LinkManager() {
//        for (int i = 0; i < 4; ++i)
//            if (Connections[i])
//                Connections[i]->Alive = false;
//    }
//};

struct ThreadPackage {
    std::thread* Array[MaxCPUThreads]{ nullptr };
    bool Occupied = false;

    ThreadPackage() = default;

    inline void Release() {
        Occupied = false;
    }
};

class ResourceManager
    // Class used to gloabaly collect info about resource usage probably used in future
{
    static unsigned short ExistingInstances;
    static unsigned long AllUsedMemory;
    static const unsigned short ThreadsAssetsSetsAmount = 16;
    static ThreadPackage ThreadAssets[ThreadsAssetsSetsAmount];
public:

    static ThreadPackage& GetThreads() {
        for (int i = 0; i < ThreadsAssetsSetsAmount; ++i) {
            if (!ThreadAssets[i].Occupied) {
                ThreadAssets[i].Occupied = true;
                return ThreadAssets[i];
            }
        }
#ifdef _MSC_VER 
        throw std::exception("Thread resources overloaded");
#else
        throw std::exception();
#endif  
    }


protected:
    unsigned long UsedMemory = 0; //MB

    const short MemoryAssetsSize = 64;
    const short OperatingRegionSize = 128;
    Region* MemoryAssets[64]{nullptr};
    unsigned short UsedRegions = 0;
    unsigned short MemoryAssetsInd = 0;

    void SortMemoryAssets();

public:
    ResourceManager();
    ~ResourceManager();

    Region* GetRegion(unsigned);
    void DeleteRegion(Region*);

    template<typename T>
    void ArrayPop(T** ArrayPtr, unsigned long long ArraySize) {
        MemoryAssets[0]->Allocate(ArrayPtr, ArraySize);
    }
};

//void ResourceManagerThread() {
//
//}

#endif