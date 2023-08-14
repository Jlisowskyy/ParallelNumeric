// Author: Jakub Lisowski
#include "../Include/Wrappers/ParallelNumeric.hpp"

#ifdef OpSysWIN_
 
int FindConsoleWidth(){
	CONSOLE_SCREEN_BUFFER_INFO buff;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &buff);

	return buff.srWindow.Right - buff.srWindow.Left + 1;
}


#elif defined __OpSysUNIX__

#endif
