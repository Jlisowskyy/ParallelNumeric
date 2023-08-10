
// Author: Jakub Lisowski

#ifndef PARALLELNUM_DEBUGERS_H_
#define PARALLELNUM_DEBUGERS_H_

#include <chrono>
#include <iostream>

class Timer {
	std::chrono::steady_clock::time_point begin;
	const char* TimerName;
	const bool WriteOnDeath;
	bool IsComputingAverage = false;
	bool VerboseAverage = false;
	unsigned RunsToIncludeInAverage = 0;
	unsigned RunsDone = 0;
	long long RunsSum = 0;
	long long LastRun;
	const unsigned TimerID;
	static unsigned TimerCount;

	void WriteMessage(long long SpentTime) {
		std::cout << "\nTimer with ID: " << TimerID;

		if (TimerName) std::cout << ", and name: " << TimerName;
			
		std::cout << ", spent time: \nIn seconds: " << SpentTime * 1e-9 <<
			"\nIn miliseconds: " << SpentTime * 1e-6 << "\nIn microseconds: " <<
			SpentTime * 1e-3 << "\nIn nanoseconds: " << SpentTime << std::endl;
	}

	void ShortenAverageToThisRun() {
		RunsToIncludeInAverage = RunsDone + 1;
	}

public:
	Timer(const char* TimerName = nullptr, bool WOD = true) :
		WriteOnDeath{ WOD }, TimerID{ TimerCount++ }, TimerName{ TimerName }
	{
		Start();
	}
	
	void Start() {
		begin = std::chrono::steady_clock::now();
	}

	long long Stop() {
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

	void Stop_and_Start() {
		Stop();
		Start();
	}

	void CalculateAverageTime(unsigned AmountOfTries, bool Verbose = false) {
		if (AmountOfTries == 0) {
			std::cerr << "[ERROR] Timer enocuntered zero division in Average function\n";
			return;
		}

		IsComputingAverage = true;
		RunsToIncludeInAverage = AmountOfTries;
		RunsDone = 0;
		RunsSum = 0;
		VerboseAverage = Verbose;
		Start();
	}

	void InvalidateLastRun() {
		if (RunsDone == 0) return;

		--RunsDone;
		RunsSum -= LastRun;
	}

	~Timer() {
		if (WriteOnDeath) {
			ShortenAverageToThisRun();
			Stop();
		}
	}
};

void AL() {
	std::cout << "!!!";
}

template<typename T>
class DebuggerFoundation {
protected:
	const unsigned ObjectID;
	static unsigned EntityCounter;
public:
	DebuggerFoundation():
		ObjectID{ EntityCounter++ }
	{
		
	}

	~DebuggerFoundation()
	{

	}
};

template<typename T>
unsigned DebuggerFoundation<T>::EntityCounter = 0;
unsigned Timer::TimerCount = 0;

#endif