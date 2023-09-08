//
// Created by Jlisowskyy on 13/08/2023.
//

#include "../Include/Operations/NumericalCore.hpp"
#include "../Include/Operations/AVXSolutions.hpp"

// -----------------------------------------
// AVX specialisations
// -----------------------------------------

// -----------------------------------------
// Matrix sum
// -----------------------------------------

#ifdef __AVX__

// TODO Make universal template

template<>
void MatrixSumMachine<double>::AlignedArrays(size_t Begin, size_t End){
    for (size_t i = Begin; i < End; i += GetCacheLineElem<double>()) {
        _mm256_store_pd(MatC + i, _mm256_add_pd(
                _mm256_load_pd(MatA + i),
                _mm256_load_pd(MatB + i)
        ));
        _mm256_store_pd(MatC + i + AVXInfo::f64Cap, _mm256_add_pd(
                _mm256_load_pd(MatA + i + AVXInfo::f64Cap),
                _mm256_load_pd(MatB + i + AVXInfo::f64Cap)
        ));
    }
}

template<>
void MatrixSumMachine<float>::AlignedArrays(size_t Begin, size_t End) {
    for (size_t i = Begin; i < End; i += AVXInfo::f32Cap) {
        _mm256_store_ps(MatC + i, _mm256_add_ps(
                _mm256_load_ps(MatA + i),
                _mm256_load_ps(MatB + i)
        ));
        _mm256_store_ps(MatC + i + AVXInfo::f32Cap, _mm256_add_ps(
                _mm256_load_ps(MatA + i + AVXInfo::f32Cap),
                _mm256_load_ps(MatB + i + AVXInfo::f32Cap)
        ));
    }
}

#endif // __AVX__

// ------------------------------------------
// Dot product
// ------------------------------------------

#ifdef __AVX__

template<>
double DotProductMachine<double>::DotProductAligned(size_t Begin, size_t End)
    // Highly limited by memory bandwidth
{
    using AVXInfo::f64Cap;
    static constexpr size_t LoadersCount = 3;

    __m256d AccRegs[AVXAccumulators] {_mm256_setzero_pd() };
    __m256d LoadRegs[LoadersCount] {_mm256_setzero_pd() };

    auto ApplyFMA = [&](size_t AccInd, size_t LoadInd, size_t Iter) -> void
        // Executes single operation on Chosen accumulator and Loader registers with respect to passed Index in data
    {
        AccRegs[AccInd] = _mm256_fmadd_pd(
                LoadRegs[LoadInd],
                _mm256_load_pd(BPtr + Iter + AccInd * f64Cap),
                AccRegs[AccInd]
        );
    };

    auto ProcessLoadAndApply = [&](size_t FirstAccInd, size_t Iter)
        // Does single blocked load operation and fma operation also
    {
        LoadRegs[0] = _mm256_load_pd(APtr + Iter + FirstAccInd * f64Cap);
        LoadRegs[1] = _mm256_load_pd(APtr + Iter + (FirstAccInd + 1) * f64Cap);
        LoadRegs[2] = _mm256_load_pd(APtr + Iter + (FirstAccInd + 2) * f64Cap);

        ApplyFMA(FirstAccInd, 0, Iter);
        ApplyFMA(FirstAccInd + 1, 1, Iter);
        ApplyFMA(FirstAccInd + 2, 2, Iter);
    };


    for (size_t i = Begin; i < End; i+=GetKernelSize())
        // Accumulates values to 12 registers
    {
        ProcessLoadAndApply(0,i);
        ProcessLoadAndApply(3,i);
        ProcessLoadAndApply(6,i);
        ProcessLoadAndApply(9,i);
    }

    AccRegs[0] += AccRegs[1];
    AccRegs[2] += AccRegs[3];
    AccRegs[4] += AccRegs[5];
    AccRegs[6] += AccRegs[7];
    AccRegs[8] += AccRegs[9];
    AccRegs[10] += AccRegs[11];

    AccRegs[0] += AccRegs[2];
    AccRegs[4] += AccRegs[6];
    AccRegs[8] += AccRegs[10];

    AccRegs[0] += AccRegs[4];
    AccRegs[0] += AccRegs[8];

    return HorSum(AccRegs[0]);
}

#endif // __AVX__

// ------------------------------------------
// Outer product AVX Implementation
// ------------------------------------------

#ifdef __AVX__

template<>
void OuterProductMachine<double>::ProcessCoefBlock(size_t BlockBegin, size_t BlockEnd, size_t VectRange){
    const double *CoefPtrIter = CoefPtr;
    for (size_t i = BlockBegin; i < BlockEnd; i += GetCacheLineElem<double>()) {
        __m256d CoefBuff0 = _mm256_set1_pd(*(CoefPtrIter));
        __m256d CoefBuff1 = _mm256_set1_pd(*(CoefPtrIter + 1));
        __m256d CoefBuff2 = _mm256_set1_pd(*(CoefPtrIter + 2));
        __m256d CoefBuff3 = _mm256_set1_pd(*(CoefPtrIter + 3));
        __m256d CoefBuff4 = _mm256_set1_pd(*(CoefPtrIter + 4));
        __m256d CoefBuff5 = _mm256_set1_pd(*(CoefPtrIter + 5));
        __m256d CoefBuff6 = _mm256_set1_pd(*(CoefPtrIter + 6));
        __m256d CoefBuff7 = _mm256_set1_pd(*(CoefPtrIter + 7));
        CoefPtrIter += GetCacheLineElem<double>();

        const double *VectPtrIter = VectPtr;
        for (size_t j = 0; j < VectRange; j += GetCacheLineElem<double>()) {
            __m256d VectA0 = _mm256_load_pd(VectPtrIter);
            __m256d VectA1 = _mm256_load_pd(VectPtrIter + AVXInfo::f64Cap);
            VectPtrIter += GetCacheLineElem<double>();

            double *TargetFirstPtr0 = MatC + i * MatCSoL + j;
            double *TargetSecondPtr0 = MatC + i * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr1 = MatC + (i + 2) * MatCSoL + j;
            double *TargetSecondPtr1 = MatC + (i + 2) * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr2 = MatC + (i + 4) * MatCSoL + j;
            double *TargetSecondPtr2 = MatC + (i + 4) * MatCSoL + j + AVXInfo::f64Cap;
            double *TargetFirstPtr3 = MatC + (i + 6) * MatCSoL + j;
            double *TargetSecondPtr3 = MatC + (i + 6) * MatCSoL + j + AVXInfo::f64Cap;

            _mm256_stream_pd(TargetFirstPtr0, _mm256_mul_pd(VectA0, CoefBuff0));
            _mm256_stream_pd(TargetSecondPtr0, _mm256_mul_pd(VectA1, CoefBuff0));
            _mm256_stream_pd(TargetFirstPtr1, _mm256_mul_pd(VectA0, CoefBuff2));
            _mm256_stream_pd(TargetSecondPtr1, _mm256_mul_pd(VectA1, CoefBuff2));
            _mm256_stream_pd(TargetFirstPtr2, _mm256_mul_pd(VectA0, CoefBuff4));
            _mm256_stream_pd(TargetSecondPtr2, _mm256_mul_pd(VectA1, CoefBuff4));
            _mm256_stream_pd(TargetFirstPtr3, _mm256_mul_pd(VectA0, CoefBuff6));
            _mm256_stream_pd(TargetSecondPtr3, _mm256_mul_pd(VectA1, CoefBuff6));

            TargetSecondPtr0 += MatCSoL;
            TargetFirstPtr0 += MatCSoL;
            TargetFirstPtr1 += MatCSoL;
            TargetSecondPtr1 += MatCSoL;
            TargetFirstPtr2 += MatCSoL;
            TargetSecondPtr2 += MatCSoL;
            TargetFirstPtr3 += MatCSoL;
            TargetSecondPtr3 += MatCSoL;

            _mm256_stream_pd(TargetFirstPtr0, _mm256_mul_pd(VectA0, CoefBuff1));
            _mm256_stream_pd(TargetSecondPtr0, _mm256_mul_pd(VectA1, CoefBuff1));
            _mm256_stream_pd(TargetFirstPtr1, _mm256_mul_pd(VectA0, CoefBuff3));
            _mm256_stream_pd(TargetSecondPtr1, _mm256_mul_pd(VectA1, CoefBuff3));
            _mm256_stream_pd(TargetFirstPtr2,_mm256_mul_pd(VectA0, CoefBuff5));
            _mm256_stream_pd(TargetSecondPtr2, _mm256_mul_pd(VectA1, CoefBuff5));
            _mm256_stream_pd(TargetFirstPtr3, _mm256_mul_pd(VectA0, CoefBuff7));
            _mm256_stream_pd(TargetSecondPtr3, _mm256_mul_pd(VectA1, CoefBuff7));
        }

        if (VectRange != VectSize){
            for (size_t ii = i; ii < i + GetCacheLineElem<double>(); ++ii) {
                for (size_t j = VectRange; j < VectSize; ++j) {
                    MatC[ii * MatCSoL + j] = VectPtr[j] * CoefPtr[ii];
                }
            }
        }
    }
}

template<>
void OuterProductMachine<double>::CleanEdges(size_t CleanBegin, size_t CleanOutElementsBegin){
    // CleaningRange (CoefSize - CleanBegin) has to be size smaller than 8, because blockable parts should be done
    // with ProcessBlock function instead of this one
    const size_t CleaningRange = CoefSize - CleanBegin;

    __m256d Buffers[GetCacheLineElem<double>()];
    for (size_t i = 0; i < CleaningRange; i++) {
        Buffers[i] = _mm256_set1_pd(CoefPtr[CleanBegin + i]);
    }

    for (size_t j = 0; j < CleanOutElementsBegin; j += GetCacheLineElem<double>()) {
        __m256d VectFirst = _mm256_load_pd(VectPtr + j);
        __m256d VectSecond = _mm256_load_pd(VectPtr + j + AVXInfo::f64Cap);

        for (size_t i = 0; i < CleaningRange; i++){
            double* CleaningTargetUpper = MatC + (i + CleanBegin) * MatCSoL + j;
            double* CleaningTargetLower = MatC + (i + CleanBegin) * MatCSoL + j + AVXInfo::f64Cap;
            _mm256_stream_pd(CleaningTargetUpper, _mm256_mul_pd(VectFirst, Buffers[i]));
            _mm256_stream_pd(CleaningTargetLower, _mm256_mul_pd(VectSecond, Buffers[i]));
        }
    }

    for (size_t i = CleanBegin; i < CoefSize; ++i) {
        for (size_t j = CleanOutElementsBegin; j < VectSize; ++j) {
            MatC[i * MatCSoL + j] = VectPtr[j] * CoefPtr[i];
        }
    }
}

#endif // __AVX__

// ------------------------------------------
// Matrix x Vector Multiplication AVX Implementation
// ------------------------------------------

#if defined(__AVX__) && defined(__FMA__)

template<>
void VMM<double>::RMVKernel8x4(const size_t HorizontalCord, const size_t VerticalCord){
    __m256d ResVectBuff0 = _mm256_setzero_pd();
    __m256d ResVectBuff1 = _mm256_setzero_pd();
    __m256d ResVectBuff2 = _mm256_setzero_pd();
    __m256d ResVectBuff3 = _mm256_setzero_pd();
    __m256d ResVectBuff4 = _mm256_setzero_pd();
    __m256d ResVectBuff5 = _mm256_setzero_pd();
    __m256d ResVectBuff6 = _mm256_setzero_pd();
    __m256d ResVectBuff7 = _mm256_setzero_pd();

    const size_t Range = std::min(HorizontalCord + GetVectChunkSize(), MatACols);
    for(size_t i = HorizontalCord; i < Range; i += 4)
        #define LoadVectPart(offset) OmitCacheLoad(RefPtr + (VerticalCord + offset) * MatASoL)
    {
        const double* RefPtr = MatA + i;
        __m256d CoefBuff = _mm256_load_pd(VectB + i);

        ResVectBuff0 = _mm256_fmadd_pd(LoadVectPart(0), CoefBuff, ResVectBuff0);
        ResVectBuff1 = _mm256_fmadd_pd(LoadVectPart(1), CoefBuff, ResVectBuff1);
        ResVectBuff2 = _mm256_fmadd_pd(LoadVectPart(2), CoefBuff, ResVectBuff2);
        ResVectBuff3 = _mm256_fmadd_pd(LoadVectPart(3), CoefBuff, ResVectBuff3);
        ResVectBuff4 = _mm256_fmadd_pd(LoadVectPart(4), CoefBuff, ResVectBuff4);
        ResVectBuff5 = _mm256_fmadd_pd(LoadVectPart(5), CoefBuff, ResVectBuff5);
        ResVectBuff6 = _mm256_fmadd_pd(LoadVectPart(6), CoefBuff, ResVectBuff6);
        ResVectBuff7 = _mm256_fmadd_pd(LoadVectPart(7), CoefBuff, ResVectBuff7);
    }
    #undef LoadVectPart
    VectC[VerticalCord] += HorSum(ResVectBuff0);
    VectC[VerticalCord + 1] += HorSum(ResVectBuff1);
    VectC[VerticalCord + 2] += HorSum(ResVectBuff2);
    VectC[VerticalCord + 3] += HorSum(ResVectBuff3);
    VectC[VerticalCord + 4] += HorSum(ResVectBuff4);
    VectC[VerticalCord + 5] += HorSum(ResVectBuff5);
    VectC[VerticalCord + 6] += HorSum(ResVectBuff6);
    VectC[VerticalCord + 7] += HorSum(ResVectBuff7);
}



template<>
void VMM<double>::CMVKernel12x4(size_t HorizontalCord, size_t VerticalCord){

}

#endif