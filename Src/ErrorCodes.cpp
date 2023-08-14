//
// Created by Jlisowskyy on 14/08/2023.
//
#include "../Include/Maintenance/ErrorCodes.hpp"

void AbandonIfNull(void* ptr){
    if (ptr == nullptr) {
#ifdef DEBUG_
        std::cerr << "[ERROR] Returned pointer was null!";
#endif
        exit(0xfa);
    }
}


