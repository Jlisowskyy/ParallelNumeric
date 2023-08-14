
// Author: Jakub Lisowski

#ifndef PARALLELNUM_DEBUGGERS_H_
#define PARALLELNUM_DEBUGGERS_H_

#include <chrono>

class Timer {
	std::chrono::steady_clock::time_point begin;
	const char* TimerName;
	const bool WriteOnDeath;
	bool IsComputingAverage = false;
	bool VerboseAverage = false;
	unsigned RunsToIncludeInAverage = 0;
	unsigned RunsDone = 0;
	long long RunsSum = 0;
	long long LastRun = 0;
	const unsigned TimerID;
	static unsigned TimerCount;

	void WriteMessage(long long SpentTime);

	void ShortenAverageToThisRun() {
		RunsToIncludeInAverage = RunsDone + 1;
	}

public:
	Timer(const char* TimerName = nullptr, bool WOD = true) :
        TimerName{ TimerName }, WriteOnDeath{ WOD }, TimerID{ TimerCount++ }
	{
		Start();
	}
	
	void Start() { begin = std::chrono::steady_clock::now(); }
    long long Stop();
    void CalculateAverageTime(unsigned AmountOfTries, bool Verbose = false);

	void Stop_and_Start() {
		Stop();
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

#endif