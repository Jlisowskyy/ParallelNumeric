
// Author: Jakub Lisowski

#ifndef PARALLELNUM_ERRORS_H_
#define PARALLELNUM_ERRORS_H_

#include <cstdlib>
#include <iostream>
#include "../Wrappers/ParallelNumeric.hpp"

void AbandonIfNull(void* ptr) {
	if (ptr == nullptr) {
#ifdef DEBUG_
		std::cerr << "[ERROR] Returned pointer was null!";
#endif
		exit(0xfa);
	}
}

#endif