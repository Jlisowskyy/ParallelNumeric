//
// Created by Jlisowskyy on 21/08/2023.
//

#ifndef PARALLELNUM_MATRIXMULTIPLICATION_HPP
#define PARALLELNUM_MATRIXMULTIPLICATION_HPP

#include "../Management/ResourceManager.hpp"
#include <iostream>

using cun = const unsigned;
using ul = unsigned long;
using cul = const unsigned long;
using ull = unsigned long long;
using cull = const unsigned long long;

template<typename NumType>
class GPMM
        // General purpose matrix multiplication
        // Performs, optimized for available cache sizes, operation C += A * B
        //
        //                               Dim3
        //                             -------->
        //                    Dim2 |   _________
        //                         |  |         |
        //               Dim2     \|/ |  MatB   |
        //              -------->  '  |_________|
        //              _________      _________
        //          ^  |         |    |         | <--- Positions refers to indexes on C matrix
        //     Dim1 |  |  MatA   |    |  MatC   |
        //          |  |_________|    |_________|
        // Not fully optimized for short A matrix
{
    // Chosen blocking parameters for specific cpu attributes
    static constexpr size_t Dim1Part = 12240;
    static constexpr size_t Dim2Part = 240;
    static constexpr size_t Dim3Part = 1020;
    static constexpr size_t HorInBlockSize = 6;

    // Matrices parameters
    const NumType* const MatA;
    const NumType* const MatB;
    NumType* const MatC;
    size_t Dim1, Dim2, Dim3;
    size_t MatASoL, MatBSoL, MatCSoL; // Size of single line necessary, caused by applied alignment

public:
    GPMM(const NumType* MatAData, const NumType* MatBData, NumType* MatCData,
         size_t Dim1, size_t Dim2, size_t Dim3,
         size_t MatASizeOfLine, size_t MatBSizeOfLine, size_t MatCSizeOfLine);

    void CCPerform(){
        std::cerr << "[ERROR] GENERAL PERFORM NOT IMPLEMENTED YET\n";
    } // TODO

private:
    inline void CCKernelXx6(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off){
        std::cerr << "[ERROR] GENERAL KERNELXx6 NOT IMPLEMENTED YET\n";
    } // TODO

    inline void CCKernelXxY(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off, size_t HorKernelSize){
        std::cerr << "[ERROR] GENERAL KERNELXxY NOT IMPLEMENTED YET\n";
    }

    inline void CCInnerParts(size_t VerOut, size_t HorOut, size_t Dim2Outer){
        std::cerr << "[ERROR] GENERAL INNER PARTS NOT IMPLEMENTED YET\n";
    }

};

//--------------------------------------
// AVX / FMA SPEC
//--------------------------------------

#if defined(__AVX__) && defined(__FMA__)

template<>
inline void GPMM<double>::CCKernelXx6(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off);

template<>
inline void GPMM<double>::CCKernelXxY(size_t HorizontalCord, size_t VerticalCord, size_t Dim2Off, size_t HorKernelSize);

template<>
inline void GPMM<double>::CCInnerParts(size_t VerOut, size_t HorOut, size_t Dim2Outer);

template<>
void GPMM<double>::CCPerform();

#endif

//--------------------------------------
// Implementation
//--------------------------------------

template<typename NumType>
GPMM<NumType>::GPMM(const NumType *MatAData, const NumType *MatBData, NumType *MatCData, size_t Dim1,
                    size_t Dim2, size_t Dim3, size_t MatASizeOfLine, size_t MatBSizeOfLine,
                    size_t MatCSizeOfLine):
        MatA { MatAData }, MatB{ MatBData }, MatC{ MatCData }, Dim1{ Dim1 }, Dim2{ Dim2 }, Dim3{ Dim3 },
        MatASoL{ MatASizeOfLine }, MatBSoL{ MatBSizeOfLine }, MatCSoL { MatCSizeOfLine }
    // Decides which approach is the most optimal for this current case
{

}

#endif //PARALLELNUM_MATRIXMULTIPLICATION_HPP