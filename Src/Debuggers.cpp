//
// Created by Jlisowskyy on 13/08/2023.
//

#include <iostream>
#include "../Include/Maintenance/Debuggers.hpp"

unsigned Timer::TimerCount = 0;

// --------------------------------------------
// class life manipulation implementation
// --------------------------------------------

Timer::~Timer() {
    if (WriteOnDeath) {
        ShortenAverageToThisRun();
        Stop();
    }
}

Timer::Timer(const char *YourName, bool WriteOnDeath) :
        TimerName{ YourName }, WriteOnDeath{ WriteOnDeath }, TimerID{TimerCount++ }
{
    Start();
}

// --------------------------------------
// class interaction implementation
// --------------------------------------

long long Timer::Stop()  {
    std::chrono::steady_clock::duration Duration =
            std::chrono::steady_clock::now().time_since_epoch() - begin.time_since_epoch();

    long long DurationVal = Duration.count();

    if (IsComputingAverage) {
        ++RunsDone;
        RunsSum += DurationVal;
        LastRun = DurationVal;

        if (VerboseAverage) {
            std::cout << "Run number: " << RunsDone << '\n';
            WriteMessage(DurationVal);
        }

        if (RunsDone == RunsToIncludeInAverage) {
            IsComputingAverage = false;

            std::cout << "\nAverage over " << RunsDone << " has been computed:\n";
            WriteMessage(RunsSum / RunsDone);
        }
    }
    else {
        WriteMessage(DurationVal);
    }

    return DurationVal;
}

void Timer::CalculateAverageTime(unsigned int AmountOfTries, bool Verbose) {
    if (AmountOfTries == 0) {
        std::cerr << "[ERROR] Timer encountered zero division in Average function\n";
        return;
    }

    IsComputingAverage = true;
    RunsToIncludeInAverage = AmountOfTries;
    RunsDone = 0;
    RunsSum = 0;
    VerboseAverage = Verbose;
    Start();
}

void Timer::Stop_and_Start() {
    Stop();
    Start();
}



void Timer::InvalidateLastRun() {
    if (RunsDone == 0) return;

    --RunsDone;
    RunsSum -= LastRun;
}

void Timer::Start() { begin = std::chrono::steady_clock::now(); }



// ------------------------------------
// private methods implementation
// ------------------------------------

void Timer::WriteMessage(long long SpentTime){
    std::cout << "\nTimer with ID: " << TimerID;

    if (TimerName) std::cout << ", and name: " << TimerName;

    std::cout << ", spent time: \nIn seconds: " << (double) SpentTime * 1e-9 <<
        "\nIn milliseconds: " << (double) SpentTime * 1e-6 << "\nIn microseconds: " <<
        (double) SpentTime * 1e-3 << "\nIn nanoseconds: " << SpentTime << std::endl;
}

void Timer::ShortenAverageToThisRun() {
    RunsToIncludeInAverage = RunsDone + 1;
}