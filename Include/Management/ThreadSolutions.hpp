//
// Created by Jlisowskyy on 05/09/2023.
//

#ifndef PARALLELNUMERIC_THREADSOLUTIONS_HPP
#define PARALLELNUMERIC_THREADSOLUTIONS_HPP

#include <cstdlib>
#include "ResourceManager.hpp"

template<typename NumType, typename... ThreadParam>
inline void ExecuteThreadsWOutput(NumType* ResultTab, size_t ThreadAmount, ThreadParam... Args)
    // Commonly used template to use with the task into parts breaking thread solutions
    // Executes threads, which saves their results into ResultTab with respect to their birth order
{
    ThreadPackage& Threads = ResourceManager::GetThreads();
    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i] = new std::thread(Args..., i, ResultTab + i);
    }

    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i]->join();
        delete Threads.Array[i];
    }
    Threads.Release();
}

template<typename... ThreadParam>
inline void ExecuteThreads(size_t ThreadAmount, ThreadParam... Args)
// Commonly used template to use with the task into parts breaking thread solutions
// Executes threads, with assumptions that only thread input is its birth order symbol
{
    ThreadPackage& Threads = ResourceManager::GetThreads();
    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i] = new std::thread(Args..., i);
    }

    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i]->join();
        delete Threads.Array[i];
    }
    Threads.Release();
}

template<typename... ThreadParam>
inline ThreadPackage& ExecuteThreadsWNJoining(size_t ThreadAmount, ThreadParam... Args)
    // Just spawns thread without id inputs, should be joined after
{
    ThreadPackage& Threads = ResourceManager::GetThreads();
    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i] = new std::thread(Args...);
    }
    return Threads;
}

inline void JoinThreads(size_t ThreadAmount, ThreadPackage&  Threads)
    // Joins previously started ThreadPackage
{
    for (size_t i = 0; i < ThreadAmount; ++i) {
        Threads.Array[i]->join();
        delete Threads.Array[i];
    }
    Threads.Release();
}

#endif //PARALLELNUMERIC_THREADSOLUTIONS_HPP
